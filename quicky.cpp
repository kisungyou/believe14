// run with > c++ -std=c++14 -I./include quicky.cpp -o quicky
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <believe14.hpp>

namespace {

struct Dataset {
  Eigen::MatrixXd X;
  std::vector<std::string> labels;
};

struct RGB {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    tokens.push_back(token);
  }
  return tokens;
}

bool looks_like_number(const std::string& s) {
  if (s.empty()) {
    return false;
  }
  char* end = nullptr;
  std::strtod(s.c_str(), &end);
  return end != nullptr && *end == '\0';
}

Dataset load_iris_csv(const std::string& path) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  std::vector<std::array<double, 4>> rows;
  std::vector<std::string> labels;
  std::string line;
  bool seen_data = false;

  while (std::getline(fin, line)) {
    if (line.empty()) {
      continue;
    }
    auto tokens = split_csv_line(line);
    if (tokens.size() < 5) {
      continue;
    }

    if (!looks_like_number(tokens[0])) {
      if (!seen_data) {
        continue;  // header row
      }
      throw std::runtime_error("Invalid numeric value in data row: " + line);
    }

    std::array<double, 4> r{};
    for (int i = 0; i < 4; ++i) {
      r[static_cast<std::size_t>(i)] = std::stod(tokens[static_cast<std::size_t>(i)]);
    }
    rows.push_back(r);
    labels.push_back(tokens[4]);
    seen_data = true;
  }

  if (rows.empty()) {
    throw std::runtime_error("No usable rows found in: " + path);
  }

  Eigen::MatrixXd X(static_cast<Eigen::Index>(rows.size()), 4);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(rows.size()); ++i) {
    for (Eigen::Index j = 0; j < 4; ++j) {
      X(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }

  return Dataset{X, labels};
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= static_cast<std::uint32_t>(data[i]);
    for (int k = 0; k < 8; ++k) {
      const std::uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

std::uint32_t adler32(const std::uint8_t* data, std::size_t size) {
  const std::uint32_t mod = 65521u;
  std::uint32_t a = 1u;
  std::uint32_t b = 0u;
  for (std::size_t i = 0; i < size; ++i) {
    a = (a + data[i]) % mod;
    b = (b + a) % mod;
  }
  return (b << 16) | a;
}

void append_u32_be(std::vector<std::uint8_t>& out, std::uint32_t x) {
  out.push_back(static_cast<std::uint8_t>((x >> 24) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>((x >> 16) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>((x >> 8) & 0xFFu));
  out.push_back(static_cast<std::uint8_t>(x & 0xFFu));
}

void append_chunk(std::vector<std::uint8_t>& png, const char type[4], const std::vector<std::uint8_t>& data) {
  append_u32_be(png, static_cast<std::uint32_t>(data.size()));
  const std::size_t type_start = png.size();
  png.push_back(static_cast<std::uint8_t>(type[0]));
  png.push_back(static_cast<std::uint8_t>(type[1]));
  png.push_back(static_cast<std::uint8_t>(type[2]));
  png.push_back(static_cast<std::uint8_t>(type[3]));
  png.insert(png.end(), data.begin(), data.end());
  const std::uint32_t crc = crc32(png.data() + type_start, 4 + data.size());
  append_u32_be(png, crc);
}

void write_png_rgb(const std::string& output_path, int width, int height, const std::vector<std::uint8_t>& rgb) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid PNG dimensions.");
  }
  if (rgb.size() != static_cast<std::size_t>(width * height * 3)) {
    throw std::runtime_error("RGB buffer size mismatch.");
  }

  std::vector<std::uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(height) * (1 + static_cast<std::size_t>(width) * 3));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);  // filter type 0
    const std::size_t row_start = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3;
    raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(row_start),
               rgb.begin() + static_cast<std::ptrdiff_t>(row_start + static_cast<std::size_t>(width) * 3));
  }

  std::vector<std::uint8_t> zlib;
  zlib.push_back(0x78);  // zlib header
  zlib.push_back(0x01);  // no compression/fastest

  std::size_t offset = 0;
  while (offset < raw.size()) {
    const std::size_t remaining = raw.size() - offset;
    const std::uint16_t block_len =
        static_cast<std::uint16_t>(std::min<std::size_t>(remaining, 65535u));
    const bool final_block = (offset + block_len == raw.size());
    zlib.push_back(final_block ? 0x01 : 0x00);
    zlib.push_back(static_cast<std::uint8_t>(block_len & 0xFFu));
    zlib.push_back(static_cast<std::uint8_t>((block_len >> 8) & 0xFFu));
    const std::uint16_t nlen = static_cast<std::uint16_t>(~block_len);
    zlib.push_back(static_cast<std::uint8_t>(nlen & 0xFFu));
    zlib.push_back(static_cast<std::uint8_t>((nlen >> 8) & 0xFFu));
    zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                raw.begin() + static_cast<std::ptrdiff_t>(offset + block_len));
    offset += block_len;
  }

  append_u32_be(zlib, adler32(raw.data(), raw.size()));

  std::vector<std::uint8_t> png;
  const std::uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  png.insert(png.end(), signature, signature + 8);

  std::vector<std::uint8_t> ihdr;
  append_u32_be(ihdr, static_cast<std::uint32_t>(width));
  append_u32_be(ihdr, static_cast<std::uint32_t>(height));
  ihdr.push_back(8);  // bit depth
  ihdr.push_back(2);  // color type RGB
  ihdr.push_back(0);  // compression
  ihdr.push_back(0);  // filter
  ihdr.push_back(0);  // interlace
  append_chunk(png, "IHDR", ihdr);
  append_chunk(png, "IDAT", zlib);
  append_chunk(png, "IEND", {});

  std::ofstream fout(output_path, std::ios::binary);
  if (!fout.is_open()) {
    throw std::runtime_error("Could not write PNG: " + output_path);
  }
  fout.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
}

void set_pixel(std::vector<std::uint8_t>& rgb, int width, int height, int x, int y, const RGB& color) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  const std::size_t idx = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                           static_cast<std::size_t>(x)) *
                          3;
  rgb[idx + 0] = color.r;
  rgb[idx + 1] = color.g;
  rgb[idx + 2] = color.b;
}

void draw_line(std::vector<std::uint8_t>& rgb,
               int width,
               int height,
               int x0,
               int y0,
               int x1,
               int y1,
               const RGB& color) {
  int dx = std::abs(x1 - x0);
  int sx = x0 < x1 ? 1 : -1;
  int dy = -std::abs(y1 - y0);
  int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;

  while (true) {
    set_pixel(rgb, width, height, x0, y0, color);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

void draw_filled_circle(std::vector<std::uint8_t>& rgb,
                        int width,
                        int height,
                        int cx,
                        int cy,
                        int radius,
                        const RGB& color) {
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx * dx + dy * dy <= radius * radius) {
        set_pixel(rgb, width, height, cx + dx, cy + dy, color);
      }
    }
  }
}

void write_png_scatter(const Eigen::Ref<const Eigen::MatrixXd>& embedding,
                       const std::vector<std::string>& labels,
                       const std::string& output_path) {
  if (embedding.cols() < 2) {
    throw std::runtime_error("Embedding must have at least 2 columns for 2D visualization.");
  }
  if (embedding.rows() != static_cast<Eigen::Index>(labels.size())) {
    throw std::runtime_error("Embedding row count does not match label count.");
  }

  const int width = 900;
  const int height = 650;
  const int margin = 60;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width * height * 3), 255);

  const double x_min = embedding.col(0).minCoeff();
  const double x_max = embedding.col(0).maxCoeff();
  const double y_min = embedding.col(1).minCoeff();
  const double y_max = embedding.col(1).maxCoeff();

  const double x_span = (x_max - x_min == 0.0) ? 1.0 : (x_max - x_min);
  const double y_span = (y_max - y_min == 0.0) ? 1.0 : (y_max - y_min);

  auto scale_x = [&](double x) -> int {
    const double px = margin + (x - x_min) / x_span * (width - 2 * margin);
    return static_cast<int>(std::lround(px));
  };
  auto scale_y = [&](double y) -> int {
    const double py = height - margin - (y - y_min) / y_span * (height - 2 * margin);
    return static_cast<int>(std::lround(py));
  };

  const std::vector<RGB> palette = {
      {31, 119, 180}, {214, 39, 40}, {44, 160, 44}, {148, 103, 189}, {255, 127, 14}, {23, 190, 207}};
  std::unordered_map<std::string, RGB> label_to_color;
  std::vector<std::string> unique_labels;

  for (const auto& label : labels) {
    if (label_to_color.find(label) == label_to_color.end()) {
      const std::size_t idx = unique_labels.size() % palette.size();
      label_to_color[label] = palette[idx];
      unique_labels.push_back(label);
    }
  }

  draw_line(rgb, width, height, margin, height - margin, width - margin, height - margin, {68, 68, 68});
  draw_line(rgb, width, height, margin, margin, margin, height - margin, {68, 68, 68});

  for (Eigen::Index i = 0; i < embedding.rows(); ++i) {
    const int px = scale_x(embedding(i, 0));
    const int py = scale_y(embedding(i, 1));
    const RGB& color = label_to_color[labels[static_cast<std::size_t>(i)]];
    draw_filled_circle(rgb, width, height, px, py, 4, color);
  }

  int legend_y = margin;
  const int legend_x = width - margin - 160;
  for (const auto& label : unique_labels) {
    draw_filled_circle(rgb, width, height, legend_x, legend_y, 5, label_to_color[label]);
    legend_y += 22;
  }

  write_png_rgb(output_path, width, height, rgb);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    (void)argc;
    (void)argv;
    const std::string input_csv = "iris.csv";
    const std::string output_png = "quicky-plot.png";
    const Eigen::Index ndim = 2;

    const Dataset data = load_iris_csv(input_csv);
    auto result = believe14::pca(data.X, ndim);

    const auto it = result.objects.find("embedding");
    if (it == result.objects.end()) {
      throw std::runtime_error("Algorithm result does not contain 'embedding'.");
    }
    write_png_scatter(it->second, data.labels, output_png);

    std::cout << "Loaded rows: " << data.X.rows() << ", cols: " << data.X.cols() << "\n";
    std::cout << "Computed embedding shape: " << it->second.rows() << " x " << it->second.cols() << "\n";
    std::cout << "Wrote visualization: " << output_png << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Usage: quicky\n";
    return 1;
  }
}

Public API
==========

The package root contains version, registry, protocol, and diagnostic objects.
Estimators are imported from their scientific family.

Linear
------

.. automodule:: believe14.linear
   :members:
   :imported-members:
   :exclude-members: set_fit_request

Nonlinear
---------

.. automodule:: believe14.nonlinear
   :members:
   :imported-members:
   :exclude-members: set_fit_request

Estimation
----------

.. automodule:: believe14.estimation
   :members:
   :imported-members:
   :exclude-members: set_fit_request

Neighborhoods and embedding quality
-----------------------------------

.. automodule:: believe14.neighbors
   :members:

.. automodule:: believe14.metrics
   :members:

Using a neighbor graph in a pipeline
------------------------------------

Pass ``neighbors=graph`` directly to a supported estimator's ``fit`` method.
When scikit-learn metadata routing is enabled, use the estimator's
``set_fit_request(neighbors=True)`` to request a graph through a pipeline.
Build the graph for the exact rows being fitted, including their order.
Cross-validation folds require their own training graphs.
See the `scikit-learn metadata routing guide
<https://scikit-learn.org/stable/metadata_routing.html>`_.

Registry and contracts
----------------------

.. automodule:: believe14.api
   :members:

.. automodule:: believe14.registry
   :members:

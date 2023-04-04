.. _networks:

Networks of unit operations
===========================

Unit operation models can be composed into a network or graph, in which a node represents a unit operation and an edge denotes a connection between two unit operations.
When utilized to full extent, this allows the simulation of complicated setups and processes (e.g., SMB, MCSGP). 
A more simple use case is the addition of plug flows and stirred tanks up- and downstream of a column in order to account for dead volume and additional dispersion from the tubing.

In a network, outlet ports of unit operations can be connected to any number of inlet ports of unit operations.
Even direct cycles, where an outlet port of a unit operation is connected to its own inlet, are possible.
A unit operation does not have to possess both inlet and outlet, but it has to have at least one of them.
Pseudo unit operations such as inlet and outlet serve as sources and sinks for the network.
However, the latter is not strictly required as any terminal node (i.e., a unit operation that possesses an outlet but does not have an outgoing connection) serves as a sink.

Each connection between two unit operation ports (i.e., an edge in the graph) is equipped with a volumetric flow rate that determines the mass flow from source to target port.
These flow rates are used to determine the weight of the different incoming feeds at a unit operation’s inlet port.
Some unit operations can infer their internal flow rate (e.g., interstitial velocity) from their total incoming volumetric flow rate.
In general, the mass balance at a unit operation has to be closed, except for unit operations that act as source or sink in the network and variable volume units (e.g., stirred tanks).

The network of unit operations uses “connection”-variables :math:`c_{\text{con}}` to connect the different unit operation ports with each other.
The inlet port variables :math:`c_{\text{in},n,k}` of unit operation :math:`n` are attached to :math:`c_{\text{con},n}` via

.. math::
    :label: NetworkInletConnection

    \begin{aligned}
        c_{\text{in},n,k,i} &= c_{\text{con},n,k,i}, \qquad k = 1, \dots, N_{\text{port},\text{in},n},\quad i = 1, \dots, N_{\text{comp},n}. 
    \end{aligned}

While :math:`N_{\text{port},\text{in},n}` denotes the number of inlet ports of unit operation :math:`n`, the number of outlet ports is given by :math:`N_{\text{port},\text{out},n}`.
The connection variables :math:`c_{\text{con},n,k,i}` collect all inflows of component :math:`i` into port :math:`k` of unit operation :math:`n`: 

.. math::
    :label: NetworkConnection

    \begin{aligned}
        c_{\text{con},n,k,i} &= \frac{\sum_{m=1}^{N_{\text{units}}} \sum_{\ell = 1}^{N_{\text{port},\text{out},n}} \sum_{j = 1}^{N_{\text{comp},m}} S_{(n,k,i),(m,\ell,j)} Q_{m,\ell} c_{\text{out},m,\ell,j}}{\sum_{m=1}^{N_{\text{units}}} \sum_{\ell=1}^{N_{\text{port},\text{out},m}} \hat{S}_{(n,k),(m,\ell)} Q_{m,\ell} }, 
    \end{aligned}

where :math:`F_{m,\ell}` denotes the volumetric flow rate from outlet port :math:`\ell` of unit operation :math:`m`, :math:`S_{(n,k,i),(m,\ell,j)} \in \{0, 1\}` is a connection matrix indicating whether component :math:`i` at outlet port :math:`k` of unit operation :math:`n` is connected to component :math:`j` at inlet port :math:`\ell` of unit operation :math:`m`, and :math:`\hat{S}_{(n,k),(m,\ell)} \in \{0, 1\}` is another connection matrix indicating whether outlet port :math:`k` of unit operation :math:`n` is connected to inlet port :math:`\ell` of unit operation :math:`m`, that is

.. math::

    \begin{aligned}
        \hat{S}_{(n,k),(m,\ell)} = \begin{cases}
            1 & \text{if } \sum_{i = 1}^{N_{\text{comp},n}} \sum_{j = 1}^{N_{\text{comp},m}} S_{(n,k,i),(m,\ell,j)} \geq 1, \\
            0 & \text{otherwise}. \end{cases}
    \end{aligned}

Note that for each unit operation the number of inlet ports may be different from the number of outlet ports.
Hence, the mass balance of a single unit operation is taken with respect to all its ports combined.

.. _MUOPNetworkConfig:

Specification of network connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The connections between the different unit operations in the network are specified by a table.
There are two table formats:

- The long format includes seven columns.
  The first two columns specify source and destination unit operation id.
  The next two columns give source and destination port indices.
  Source and destination component indices are given by the following two columns.
  Finally, the seventh column specifies the volumetric flow rate of this connection (see :ref:`FFModelConnectionSwitch`).

- The short format includes five columns.
  The first two columns specify source and destination unit operation id.
  Source and destination component indices are given by the following two columns.
  Finally, the fifth column specifies the volumetric flow rate of this connection.
  Here, the omitted port indices default to :math:`-1`, which connects all ports of the source unit operation to the corresponding ports of the target.

By default, the short format is used (i.e., a table with five columns is expected).
However, if a unit operation with multiple ports is present, a table with seven columns is required.
The default format can be overruled by setting a field.

With this setup, it is possible to connect single components of unit operations with each other yielding a maximum in flexibility.
However, the predominant case is to connect all components of the source unit operations with their respective counterparts in the destination unit.
This can easily be done by setting both component indices to :math:`-1` instead of writing a separate row for each component of the connection.
The same setting (i.e., setting both port indices to :math:`-1`) can be used to connect all ports of one unit operation with all corresponding ports of another one.

Note that in case of multiple rows for one connection between two unit operation ports (e.g., in case of separate component connections) the flow rate of the first row of that connection is used and all following flow rates are ignored.
Consequently, there can only be one flow rate for a connection between two unit operations regardless of which components are connected.

The connection table is expected in row-major storage format (i.e., the rows are appended to one long array). 

.. _MUOPNetworkValveSwitches:

Valve switches
~~~~~~~~~~~~~~

The connectivity of the network can only change on a discontinuous section transition.
Such a transition with changing connectivity is referred to as valve switch and the connectivity itself as valve configuration.

A list of valve configurations with at least one entry is required.
Each valve configuration consists of a network connectivity table as described in Section :ref:`MUOPNetworkConfig` and a section index.
The latter denotes the section in which the connectivity table becomes active.
Hence, the one required (i.e., the first) entry must have a section index of :math:`0` denoting the initial connectivity.

Note that the section index has to be monotonically increasing throughout the list of valve configurations.
See Tables :ref:`FFModelSystemConnections` and :ref:`FFModelConnectionSwitch`.


.. _MUOPNetworkDynamicFlowRates:

Dynamic flow rates
~~~~~~~~~~~~~~~~~~

The volumetric flow rates may vary over time while the valve configuration is active.
The rates are assumed to be cubic polynomials,

.. math::

      Q = Q_0 + Q_1(t - t_s) + Q_2(t-t_s)^2 + Q_3(t-t_s)^3,

where :math:`t_s` is the beginning of the time section that triggers the valve switch.

Note that the denominator in Eq. :eq:`NetworkConnection` must always be positive.
That is, the flow rate coefficients have to be chosen such that the flow into every connected inlet port is strictly positive at all times.


.. _MUOPNetworkLinearSolver:

Solution of the linear systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each time step in the simulation requires the solution of a nonlinear system Eq. :eq:`BDFNonlinSystem` (see Sec. :ref:`SimTimeIntegration`).
The nonlinear problem is solved by a Newton iteration, which, in turn, requires the solution of a linear system that essentially consists of the Jacobians of the unit operations and some coupling matrices from Eqs. :eq:`NetworkInletConnection` and :eq:`NetworkConnection`.

These linear systems are either solved in parallel or sequentially. The parallel method first solves each unit operation (in parallel) to compute the solution at its outlet.
Using these values, the inlets are adjusted and the unit operations are solved again.
This is iterated until the system is fully solved.

In contrast, the sequential method first determines an ordering of the unit operations such that each unit only receives inflow from the previous units in the ordering.
Such an ordering requires an acyclic graph of unit operations.
Finally, the linear system is solved by solving the unit operations in the ordering determined above.
Before a unit is solved, its inlet is calculated from the outlets of the previously solved units.
This means, the system is solved from system inlets to system outlets.

The parallel method works regardless of the network topology (i.e., cycles in the graph), but requires to solve each unit operation at least twice.
The sequential method solvs each unit exactly once, but is restricted to acyclic networks and works best for small graphs.
By default, CADET uses a heuristic to select an appropriate solution method.
This default can be overridden by a flag (see Table :ref:`FFModelSolver`).

The solution method is selected for each valve switch individually.
If some network configurations contain cycles, the parallel method is chosen for them regardless of the method used for the other configurations.


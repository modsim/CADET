.. _outlet_model:

Outlet
~~~~~~

A system outlet unit operation is a pseudo unit operation since there is no physical correspondence.
The outlet serves as a sink (terminal node) in the network of unit operations.
Since any terminal node in the network is a sink (see Section :ref:`networks`), outlet unit operations are not strictly necessary.
However, in some applications (e.g., SMB) only a certain fraction of a unit operation’s output is taken out of the system and the rest is recycled.
In this case, outlet unit operations are required in order to avoid unbalanced mass flow in the other unit operations.

Outlets can also be of help if the output of multiple unit operations merges together leaving the network.
Instead of manually adding the streams together in a post-processing step, the unit operations can be connected to the same outlet unit.

For information on model parameters see :ref:`outlet_config`.

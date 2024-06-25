.. _mmc_nfor_config:

MMC Nfor 2010
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MMC_NFOR**

For information on model equations, refer to :ref:`_mmc_nfor_model`.


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =======================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =======================

``MMCNFOR_KA``
   Adsorption rate constant

**Unit:** :math:`m_{MP}^{3}~m_{SP}^{-3}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MMCNFOR_KD``
   Desorption rate constant

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MMCNFOR_KP``
   Protein-salt interaction factor (dependence of activity of solved protein from salt concentration)

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MMCNFOR_KS``
   Protein-protein interaction factor (dependence of activity of solved protein from protein concentration)

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MMCNFOR_NU``
   Characteristic charges of the protein; The number of sites
   :math:`\nu` that the protein interacts with on the resin surface

**Unit: [-]**

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================


``MMCNFOR_N``
   Number of hydrophobic ligands occupied during binding

**Unit: [-]**

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================


``MMCNFOR_SIGMA``
   Steric factors for ionic binding of the protein; The number of sites :math:`\sigma` on
   the surface that are shielded by the protein and prevented from
   exchange with the salt counterions in solution

**Unit: [-]**

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MMCNFOR_S``
   Steric factor for hydrophobic binding of the protein;; The number of sites :math:`\sigma` on
   the surface that are shielded by the protein and prevented from
   exchange with the salt counterions in solution

**Unit: [-]**

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================


``SMA_LAMBDA``
   Stationary phase capacity (monovalent salt counterions); The total
   number of binding sites available on the resin surface

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``SMA_REFC0``
   Reference liquid phase concentration (optional, defaults to
   :math:`1.0`)


**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

``SMA_REFQ``
   Reference solid phase concentration (optional, defaults to
   :math:`1.0`)


**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

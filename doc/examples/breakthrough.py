# Create component system
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem()
component_system.add_component('A')


# Inlet
from CADETProcess.processModel import Inlet

inlet = Inlet(component_system, name='inlet')
inlet.flow_rate = 6.683738370512285e-8  # m^3 / s
inlet.c = [[1.0, 0, 0, 0]]  # mol / m^3

# Column

## Adsorption
from CADETProcess.processModel import Langmuir

binding_model = Langmuir(component_system, name='binding_model')
binding_model.is_kinetic = True
binding_model.adsorption_rate = [1.0, ]  # m^3 / (mol * s)   (mobile phase
binding_model.desorption_rate = [1.0, ]  # 1 / s (desorption)
binding_model.capacity = [100.0, ]  # mol / m^3   (solid phase)

## General Rate Model

from CADETProcess.processModel import GeneralRateModel

column = GeneralRateModel(component_system, name='column')
column.binding_model = binding_model

column.length = 0.014  # m
column.diameter = 0.02  # m
column.bed_porosity = 0.37  # -
column.particle_porosity = 0.75  # -
column.particle_radius = 4.5e-5  # m

column.axial_dispersion = 5.75e-8  # m^2 / s (interstitial volume)
column.film_diffusion = [6.9e-6]  # m / s
column.pore_diffusion = [7e-10, ]  # m^2 / s (mobile phase)
column.surface_diffusion = [0.0]  # m^2 / s (solid phase)

## Initial conditions
column.c = [0]
column.cp = [0]
column.q = [0]

# Outlet
from CADETProcess.processModel import Outlet

outlet = Outlet(component_system, name='outlet')

# Flow Sheet
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, product_outlet=True)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

# Process
from CADETProcess.processModel import Process

process = Process(flow_sheet, 'Langmuir Breakthrough')
process.cycle_time = 1000.0

from CADETProcess.simulator import Cadet

process_simulator = Cadet()

simulation_results = process_simulator.simulate(process)

simulation_results.solution.column.outlet.plot()

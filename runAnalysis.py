from functions import *
from tabulate import tabulate

"""
THIS PYTHON SCRIPT CAN DO THE FOLLOWING:
	- Run a thermal analysis of given .tcl files.
	- Access these files to make .tcl files for mechanical analysis.
	- Run the mechanical analysis.
	→ Keep in mind that if the code takes long, it's only because
	  of how much time it takes OpenSees to do the analyses. Every
	  other thing in this script takes at most 3 s to do.

NOTE THAT THIS SCRIPT ASUMES:
	- ε = (ε11 + ε22 + ε33) / 3.0
	- ε12 = ε23 = ε13 = 0
	- If you want to change the type of analysis done in the mechanical
	  analysis, then go into "analysisSteps()" and do it manually.

THE INPUTS THAT THIS SCRIPT REQUIRES ARE THE FOLLOWING (CHECK LINES 45 → 53):
	- runThermal, runMechanical, step, nsteps, thermalPath, mechPath, ndf, ndm,
	  coefficient of thermal expansion (α), density of the material (ρ)

SHOULD MENTION THAT IN THE SCRIPT "functions.py" THESE FUNCTIONS AREN'T FULLY AUTOMATIC:
	- constraints() : should be inputed kind of manually → line 141 in functions.py
	- materials()   : should be inputed manually → line 218 in functions.py

TO DEAL WITH THIS, YOU COULD DO THE FOLLOWING:
	- Use the STKO file for thermal analysis, make the material, write the .tcl files
	  and then copy & paste from "materials.tcl" into the function "materials()".
	- The "constraints()"" function is kind of automatic, you need to define inside
	  the function what coords you want to fix.
"""

def main():
	initialize(mechPath, thermalPath, runThermal, runMechanical, ndf, ndm)
	definitions(mechPath)
	materials(mechPath)
	sections(mechPath)
	xyz = nodes(mechPath, thermalPath)
	eleTags, conectivity = elements(mechPath, thermalPath, selfweight=-ρ*g)
	analysisSteps(mechPath, xyz, eleTags, conectivity, α, nsteps, step)
	recorder(mechPath, thermalPath, runMechanical)

if __name__ == "__main__":
	runThermal 	  = 1         # Set to 1 if you want to compute thermal analysis, 0 otherwise
	runMechanical = 1         # Set to 1 if you want to compute mechanical analysis, 0 otherwise
	step 		  = -1        # -1 is default value to get the last step from thermal analysis
	nsteps 		  = 2         # Number of steps you want to compute in mechanical analysis
	thermalPath   = "thermal" # Folder you want to access thermal analysis files
	mechPath 	  = "mech" 	  # Folder you want to store mechanical analysis files
	ndf, ndm 	  = 3, 3 	  # Degrees of freedom & the dimension of the model
	α 			  = 12e-6 	  # [1 / °C] - P.139 ConcreteMicrostructurePropertiesandMaterials (?)
	ρ 			  = 2476 	  # [kg / m³]
	g 			  = 9.8 	  # [m / s²]

	# --------------- DON'T MODIFY ANY OF THE FOLLOWING LINES --------------- #

	if runThermal or runMechanical:
		main()

	if step == -1:
		step = "Last Step"

	setup = [["Step", step, "-"],
			 ["Mech Path", mechPath, "-"],
			 ["Thermal Path", thermalPath, "-"],
			 ["ndf", ndf, "-"],
			 ["ndm", ndm, "-"],
			 ["Thermal expansion", α, "[1 / ºC]"],
			 ["Self Weight", round(ρ*g, 3), "[kg / (s² m²)]"]]

	print(tabulate(setup, tablefmt="pretty"))
	print("""\nPLEASE READ:
  I COULDN'T MANAGE TO MAKE THE MATERIAL AND THE
  CONSTRAINTS TO WORK FULLY AUTOMATICALLY. MAYBE
  IT'S BETTER JUST TO MAKE THE EXAMPLE ONCE, GET
  THESE VALUES AND THEN RUN THE ANALYSIS :) """)
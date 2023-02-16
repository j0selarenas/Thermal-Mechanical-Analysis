from numpy import array, zeros, sqrt # pip install numpy
from scipy.linalg import det 		 # pip install scipy
import h5py							 # pip install h5py
import os 							 # built-in library

def Tet10(xyz, ue):
	# Nodal coordinates:
	x1, x2, x3, x4, x5 = xyz[0, 0], xyz[1, 0], xyz[2, 0], xyz[3, 0], xyz[4, 0]
	y1, y2, y3, y4, y5 = xyz[0, 1], xyz[1, 1], xyz[2, 1], xyz[3, 1], xyz[4, 1]
	z1, z2, z3, z4, z5 = xyz[0, 2], xyz[1, 2], xyz[2, 2], xyz[3, 2], xyz[4, 2]
	
	x6, x7, x8, x9, x10 = xyz[5, 0], xyz[6, 0], xyz[7, 0], xyz[8, 0], xyz[9, 0]
	y6, y7, y8, y9, y10 = xyz[5, 1], xyz[6, 1], xyz[7, 1], xyz[8, 1], xyz[9, 1]
	z6, z7, z8, z9, z10 = xyz[5, 2], xyz[6, 2], xyz[7, 2], xyz[8, 2], xyz[9, 2]

	# Array to hold strains for each gauss point:
	εe = zeros((4, 6))

	α, β, wi = (5.0 + 3.0 * sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0, 1.0 / 4.0
	Gauss_rule = [(wi, α, β, β),
				  (wi, β, α, β),
				  (wi, β, β, α),
				  (wi, β, β, β)]

	# --------------------------------------------------------------------------------------------------------------------------------------------------------

	# Loop over the Natural coordinates:
	for idx, (wi, ζ1, ζ2, ζ3) in enumerate(Gauss_rule):
		ζ4 = 1.0 - ζ1 - ζ2 - ζ3
		# The shape functions are given by Equation 17.2 (AFEM):
		# Note that N10 and N9 are modified to account for the fact that the nodes 9 and 10 are in a different position than they should be.
		N1, N2  = ζ1 * (2.0 * ζ1 - 1.0), ζ2 * (2.0 * ζ2 - 1.0)
		N3, N4  = ζ3 * (2.0 * ζ3 - 1.0), ζ4 * (2.0 * ζ4 - 1.0)
		N5, N6  = 4.0 * ζ1 * ζ2, 4.0 * ζ2 * ζ3
		N7, N8  = 4.0 * ζ3 * ζ1, 4.0 * ζ1 * ζ4
		N10, N9 = 4.0 * ζ2 * ζ4, 4.0 * ζ3 * ζ4
		
		# ----------------------------------------------------------------------------------------------------------------------------------------------------

		# The derivatives of the shape functions with respect to the natural coordinates are given by:
		dN1_dζ1, dN2_dζ1, dN3_dζ1, dN4_dζ1, dN5_dζ1  = 4.0 * ζ1 - 1.0, 0.0, 0.0, 0.0, 4.0 * ζ2
		dN6_dζ1, dN7_dζ1, dN8_dζ1, dN10_dζ1, dN9_dζ1 = 0.0, 4.0 * ζ3, 4.0 * ζ4, 0.0, 0.0

		dN1_dζ2, dN2_dζ2, dN3_dζ2, dN4_dζ2, dN5_dζ2  = 0.0, 4.0 * ζ2 - 1.0, 0.0, 0.0, 4.0 * ζ1
		dN6_dζ2, dN7_dζ2, dN8_dζ2, dN10_dζ2, dN9_dζ2 = 4.0 * ζ3, 0.0, 0.0, 4.0 * ζ4, 0.0

		dN1_dζ3, dN2_dζ3, dN3_dζ3, dN4_dζ3, dN5_dζ3  = 0.0, 0.0, 4.0 * ζ3 - 1.0, 0.0, 0.0
		dN6_dζ3, dN7_dζ3, dN8_dζ3, dN10_dζ3, dN9_dζ3 = 4.0 * ζ2, 4.0 * ζ1, 0.0, 0.0, 4.0 * ζ4

		dN1_dζ4, dN2_dζ4, dN3_dζ4, dN4_dζ4, dN5_dζ4  = 0.0, 0.0, 0.0, 4.0 * ζ4 - 1.0, 0.0
		dN6_dζ4, dN7_dζ4, dN8_dζ4, dN10_dζ4, dN9_dζ4 = 0.0, 0.0, 4.0 * ζ1, 4.0 * ζ2, 4.0 * ζ3

		# ----------------------------------------------------------------------------------------------------------------------------------------------------

		# The following derivatives are computed to obtain the Jacobian matrix. These equations can be found in Equation 17.9 (AFEM).
		dx_dζ1 = x1 * dN1_dζ1 + x2 * dN2_dζ1 + x3 * dN3_dζ1 + x4 * dN4_dζ1 + x5 * dN5_dζ1 + x6 * dN6_dζ1 + x7 * dN7_dζ1 + x8 * dN8_dζ1 + x9 * dN9_dζ1 + x10 * dN10_dζ1
		dy_dζ1 = y1 * dN1_dζ1 + y2 * dN2_dζ1 + y3 * dN3_dζ1 + y4 * dN4_dζ1 + y5 * dN5_dζ1 + y6 * dN6_dζ1 + y7 * dN7_dζ1 + y8 * dN8_dζ1 + y9 * dN9_dζ1 + y10 * dN10_dζ1
		dz_dζ1 = z1 * dN1_dζ1 + z2 * dN2_dζ1 + z3 * dN3_dζ1 + z4 * dN4_dζ1 + z5 * dN5_dζ1 + z6 * dN6_dζ1 + z7 * dN7_dζ1 + z8 * dN8_dζ1 + z9 * dN9_dζ1 + z10 * dN10_dζ1

		dx_dζ2 = x1 * dN1_dζ2 + x2 * dN2_dζ2 + x3 * dN3_dζ2 + x4 * dN4_dζ2 + x5 * dN5_dζ2 + x6 * dN6_dζ2 + x7 * dN7_dζ2 + x8 * dN8_dζ2 + x9 * dN9_dζ2 + x10 * dN10_dζ2
		dy_dζ2 = y1 * dN1_dζ2 + y2 * dN2_dζ2 + y3 * dN3_dζ2 + y4 * dN4_dζ2 + y5 * dN5_dζ2 + y6 * dN6_dζ2 + y7 * dN7_dζ2 + y8 * dN8_dζ2 + y9 * dN9_dζ2 + y10 * dN10_dζ2
		dz_dζ2 = z1 * dN1_dζ2 + z2 * dN2_dζ2 + z3 * dN3_dζ2 + z4 * dN4_dζ2 + z5 * dN5_dζ2 + z6 * dN6_dζ2 + z7 * dN7_dζ2 + z8 * dN8_dζ2 + z9 * dN9_dζ2 + z10 * dN10_dζ2

		dx_dζ3 = x1 * dN1_dζ3 + x2 * dN2_dζ3 + x3 * dN3_dζ3 + x4 * dN4_dζ3 + x5 * dN5_dζ3 + x6 * dN6_dζ3 + x7 * dN7_dζ3 + x8 * dN8_dζ3 + x9 * dN9_dζ3 + x10 * dN10_dζ3
		dy_dζ3 = y1 * dN1_dζ3 + y2 * dN2_dζ3 + y3 * dN3_dζ3 + y4 * dN4_dζ3 + y5 * dN5_dζ3 + y6 * dN6_dζ3 + y7 * dN7_dζ3 + y8 * dN8_dζ3 + y9 * dN9_dζ3 + y10 * dN10_dζ3
		dz_dζ3 = z1 * dN1_dζ3 + z2 * dN2_dζ3 + z3 * dN3_dζ3 + z4 * dN4_dζ3 + z5 * dN5_dζ3 + z6 * dN6_dζ3 + z7 * dN7_dζ3 + z8 * dN8_dζ3 + z9 * dN9_dζ3 + z10 * dN10_dζ3
		
		dx_dζ4 = x1 * dN1_dζ4 + x2 * dN2_dζ4 + x3 * dN3_dζ4 + x4 * dN4_dζ4 + x5 * dN5_dζ4 + x6 * dN6_dζ4 + x7 * dN7_dζ4 + x8 * dN8_dζ4 + x9 * dN9_dζ4 + x10 * dN10_dζ4
		dy_dζ4 = y1 * dN1_dζ4 + y2 * dN2_dζ4 + y3 * dN3_dζ4 + y4 * dN4_dζ4 + y5 * dN5_dζ4 + y6 * dN6_dζ4 + y7 * dN7_dζ4 + y8 * dN8_dζ4 + y9 * dN9_dζ4 + y10 * dN10_dζ4
		dz_dζ4 = z1 * dN1_dζ4 + z2 * dN2_dζ4 + z3 * dN3_dζ4 + z4 * dN4_dζ4 + z5 * dN5_dζ4 + z6 * dN6_dζ4 + z7 * dN7_dζ4 + z8 * dN8_dζ4 + z9 * dN9_dζ4 + z10 * dN10_dζ4
		
		# ----------------------------------------------------------------------------------------------------------------------------------------------------

		# The Jacobian matrix can be simplified to a 3x3 matrix by subtracting the first column of J from the last three columns. Check Equation 17.15 (AFEM).
		J = array([[dx_dζ2 - dx_dζ1, dx_dζ3 - dx_dζ1, dx_dζ4 - dx_dζ1],
				   [dy_dζ2 - dy_dζ1, dy_dζ3 - dy_dζ1, dy_dζ4 - dy_dζ1],
				   [dz_dζ2 - dz_dζ1, dz_dζ3 - dz_dζ1, dz_dζ4 - dz_dζ1]])
		
		# ----------------------------------------------------------------------------------------------------------------------------------------------------

		# The determinant of the Jacobian matrix is the volume of the tetrahedron. Check Equation 17.14 (AFEM).
		det_J = det(J) / 6.0

		# If the determinant is less or equal to zero, the node numbering is wrong.
		assert det_J > 0, "The node numbering is wrong! Mapping is not invertible!"

		# ----------------------------------------------------------------------------------------------------------------------------------------------------
		
		# Equation 16.7 (AFEM).
		a1, a2 = y2 * (z4 - z3) - y3 * (z4 - z2) + y4 * (z3 - z2), -y1 * (z4 - z3) + y3 * (z4 - z1) - y4 * (z3 - z1)
		a3, a4 = y1 * (z4 - z2) - y2 * (z4 - z1) + y4 * (z2 - z1), -y1 * (z3 - z2) + y2 * (z3 - z1) - y3 * (z2 - z1)

		b1, b2 = -x2 * (z4 - z3) + x3 * (z4 - z2) - x4 * (z3 - z2), x1 * (z4 - z3) - x3 * (z4 - z1) + x4 * (z3 - z1)
		b3, b4 = -x1 * (z4 - z2) + x2 * (z4 - z1) - x4 * (z2 - z1), x1 * (z3 - z2) - x2 * (z3 - z1) + x3 * (z2 - z1)

		c1, c2 = x2 * (y4 - y3) - x3 * (y4 - y2) + x4 * (y3 - y2), -x1 * (y4 - y3) + x3 * (y4 - y1) - x4 * (y3 - y1)
		c3, c4 = x1 * (y4 - y2) - x2 * (y4 - y1) + x4 * (y2 - y1), -x1 * (y3 - y2) + x2 * (y3 - y1) - x3 * (y2 - y1)
		
		# ----------------------------------------------------------------------------------------------------------------------------------------------------

		# Assembling a matrix of the derivatives of the shape functions with respect to the natural coordinates.
		dNi_dζj = array([[dN1_dζ1, dN1_dζ2, dN1_dζ3, dN1_dζ4],
						 [dN2_dζ1, dN2_dζ2, dN2_dζ3, dN2_dζ4],
						 [dN3_dζ1, dN3_dζ2, dN3_dζ3, dN3_dζ4],
						 [dN4_dζ1, dN4_dζ2, dN4_dζ3, dN4_dζ4],
						 [dN5_dζ1, dN5_dζ2, dN5_dζ3, dN5_dζ4],
						 [dN6_dζ1, dN6_dζ2, dN6_dζ3, dN6_dζ4],
						 [dN7_dζ1, dN7_dζ2, dN7_dζ3, dN7_dζ4],
						 [dN8_dζ1, dN8_dζ2, dN8_dζ3, dN8_dζ4],
						 [dN9_dζ1, dN9_dζ2, dN9_dζ3, dN9_dζ4],
						 [dN10_dζ1, dN10_dζ2, dN10_dζ3, dN10_dζ4]])

		# ----------------------------------------------------------------------------------------------------------------------------------------------------
		
		# The following values are computed to obtain the Strain-Displacement matrix. Check Equation 17.24 (AFEM).
		qx1, qx2, qx3, qx4, qx5, qx6, qx7, qx8, qx9, qx10 = ((1.0 / det(J)) * (dNi_dζj @ array([[a1], [a2], [a3], [a4]])).T)[0]
		qy1, qy2, qy3, qy4, qy5, qy6, qy7, qy8, qy9, qy10 = ((1.0 / det(J)) * (dNi_dζj @ array([[b1], [b2], [b3], [b4]])).T)[0]
		qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9, qz10 = ((1.0 / det(J)) * (dNi_dζj @ array([[c1], [c2], [c3], [c4]])).T)[0]
		
		# ----------------------------------------------------------------------------------------------------------------------------------------------------

		# Strain-Displacement matrix. Check Equation 17.23 (AFEM).
		B = array([[qx1, 0, 0, qx2, 0, 0, qx3, 0, 0, qx4, 0, 0, qx5, 0, 0, qx6, 0, 0, qx7, 0, 0, qx8, 0, 0, qx9, 0, 0, qx10, 0, 0],
				   [0, qy1, 0, 0, qy2, 0, 0, qy3, 0, 0, qy4, 0, 0, qy5, 0, 0, qy6, 0, 0, qy7, 0, 0, qy8, 0, 0, qy9, 0, 0, qy10, 0],
				   [0, 0, qz1, 0, 0, qz2, 0, 0, qz3, 0, 0, qz4, 0, 0, qz5, 0, 0, qz6, 0, 0, qz7, 0, 0, qz8, 0, 0, qz9, 0, 0, qz10],
				   [qy1, qx1, 0, qy2, qx2, 0, qy3, qx3, 0, qy4, qx4, 0, qy5, qx5, 0, qy6, qx6, 0, qy7, qx7, 0, qy8, qx8, 0, qy9, qx9, 0, qy10, qx10, 0],
				   [0, qz1, qy1, 0, qz2, qy2, 0, qz3, qy3, 0, qz4, qy4, 0, qz5, qy5, 0, qz6, qy6, 0, qz7, qy7, 0, qz8, qy8, 0, qz9, qy9, 0, qz10, qy10],
				   [qz1, 0, qx1, qz2, 0, qx2, qz3, 0, qx3, qz4, 0, qx4, qz5, 0, qx5, qz6, 0, qx6, qz7, 0, qx7, qz8, 0, qx8, qz9, 0, qx9, qz10, 0, qx10]])
		
		# Computing strain:
		ε = B @ ue

		# Saving data to array:
		εe[idx, :] = ε

	# --------------------------------------------------------------------------------------------------------------------------------------------------------
	
	# Returning strain  components.
	return (εe)

def constraints(xyz, fixX=1, fixY=1, fixZ=1):
	"""
	Must do this manually, but only once per example.

	L, B, H
		↪ the dimensions of the volume.

	fixed_X, fixed_Y, fixed_Z
		↪ coords to be fixed for X, Y, Z axis.

	fixX, fixY, fixZ
		↪ the gdl that are fixed (0 or 1).
	"""
	L, B, H = 1.04, 0.54, 0.50
	const, nodes = "", []
	fixed_X = array([0, L])
	fixed_Y = array([])
	fixed_Z = array([0])

	def check(fixed, coord, node, nodes):
		if len(fixed) > 0:
			for fix in fixed:
				if coord == fix:
					nodes.append(node)
		return (nodes)

	for node, (X, Y, Z) in enumerate(xyz):
		n = node + 1
		nodes = check(fixed_X, X, n, nodes)
		nodes = check(fixed_Y, Y, n, nodes)
		nodes = check(fixed_Z, Z, n, nodes)

	nodes = list(set(nodes))

	for node in nodes:
		const += f"\tfix {node} {fixX} {fixY} {fixZ}\n"

	return (const)

def initialize(mechPath, thermalPath, runThermal, runMechanical, ndf, ndm):
	f = open(f"{thermalPath}/main.tcl", "r")
	data, newdata = [], []
	for line in f:
		newline = line
		if "source" in line and ".tcl" in line:
			line = f"{line[0:line.find('.tcl')]}Mechanical.tcl\n"
			line = line[:7] + f"{mechPath}/" + line[7:]
			if "thermal" in line:
				line = line.replace(f"{thermalPath}/","")
			if "thermal" not in newline:
				newline = newline[:7] + f"{thermalPath}/" + newline[7:] 
		elif "model" in line:
			line = f"model basic -ndm {ndm} -ndf {ndf}\n"
		data.append(line)
		newdata.append(newline)
	f.close()

	f = open(f"{thermalPath}/main.tcl", "w")
	for i in newdata:
		f.write(i)
	f.close()

	if runThermal:
		os.system(f"OpenSees > source {thermalPath}\\main.tcl")

		if not runMechanical:
			exit()

	f = open(f"{mechPath}/mainMechanical.tcl", "w")
	for i in data:
		f.write(i)
	f.close()

def definitions(mechPath):
	f = open(f"{mechPath}/definitionsMechanical.tcl", "w")
	f.close()

def materials(mechPath):
	f = open(f"{mechPath}/materialsMechanical.tcl", "w")
	f.write("""nDMaterial ASDConcrete3D 1 3569006745.5 0.1 \\
	-Te 0.0 8.999999999999999e-05 0.00015000000000000001 0.0008690603703530748 0.0040453019517653725 0.040453019517653725 \\
	-Ts 0.0 321210.607095 356900.67455 71380.13491000001 0.35690067454999996 0.35690067454999996 \\
	-Td 0.0 0.0 0.0 0.7698663903881087 0.9999998763998322 0.9999999973130398 \\
	-Ce 0.0 0.0005 0.0006666666666666666 0.0008333333333333333 0.001 0.0011666666666666665 0.0013333333333333333 0.0015 0.0016666666666666666 0.0018333333333333333 0.002 0.0033413468013059615 0.0038413468013059615 \\
	-Cs 0.0 1784503.37275 2293923.4451827267 2671956.7584656216 2956661.9963743226 3170647.7627911624 3328644.8138682973 3440885.1677740915 3514818.69465633 3556069.444711038 3569006.7455 356900.67455 356900.67455 \\
	-Cd 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.7640592061052669 0.891755579106595 \\
	-rho 0.0 -eta 0.0 \\
	-implex -implexAlpha 1.0 \\
	-crackPlanes 4 4 45.0 \\
	-autoRegularization 54.0\n""")
	f.close()

def sections(mechPath):
	f = open(f"{mechPath}/sectionsMechanical.tcl", "w")
	f.close()

def nodes(mechPath, thermalPath):
	f = open(f"{thermalPath}/nodes.tcl", "r")
	data, xyz = [], []

	for idx, line in enumerate(f):
		data.append(line)
		if idx >= 1:
			words = line.split(" ")[2:]
			xyz.append(list(map(float, words)))
	f.close()

	f = open(f"{mechPath}/nodesMechanical.tcl", "w")
	for i in data:
		f.write(i)
	f.close()

	return (array(xyz))

def elements(mechPath, thermalPath, selfweight=None, materialTag=1, nnodes=10):
	f = open(f"{thermalPath}/elements.tcl", "r")
	data, eleTags, conectivity = [], [], []

	for idx, line in enumerate(f):
		if "Thermal" in line:
			line = line.replace("Thermal","")

		if idx >= 2:
			words = line.split(" ")

			conectivity.append(list(map(int, words[4:14])))
			eleTags.append(words[2])

			words = words[:4+nnodes]
			if selfweight is None:
				words.append(f"{materialTag}\n")
			else:
				words.append(f"{materialTag} {0.0} {0.0} {selfweight}\n")
			words[1] = "TenNodeTetrahedron"
			line = " ".join(words)

		data.append(line)
	f.close()

	f = open(f"{mechPath}/elementsMechanical.tcl", "w")
	for i in data:
		f.write(i)
	f.close()

	return (eleTags, conectivity)

def computeStrains(xyz, conectivity, α, step=-1):
	file_name = "recorder"
	mpcoFile  = h5py.File(f"{file_name}.mpco", "r")
	data      = mpcoFile["MODEL_STAGE[1]"]["RESULTS"]["ON_NODES"]["DISPLACEMENT"]["DATA"]

	if step == -1:
		keys  = list(data.keys())[-1]
		step  = int(keys[5:])

	disp      = array(data[f"STEP_{step}"][:, :])

	# strains at each gauss point
	strains = zeros((len(conectivity), 4))

	# --------------------> ε11 ---- ε22 ---- ε33 <-------
	avg = lambda eps: α * (eps[0] + eps[1] + eps[2]) / 3.0

	for idx, ele in enumerate(conectivity):
		ele = array(ele)-1
		ε = Tet10(xyz[ele], disp[ele].flatten())
		strains[idx] = [avg(ε[0]) , avg(ε[1]) , avg(ε[2]) , avg(ε[3])]

	return (strains)

def analysisSteps(mechPath, xyz, eleTags, conectivity, α, nsteps=1, step=-1):
	initialStrains = computeStrains(xyz, conectivity, α, step=step)
	f = open(f"{mechPath}/analysis_stepsMechanical.tcl", "w")

	f.write("""

	# Time-Increment Utility Stats
	#
	# Found 1 Physical Properties
	#    [1] ASDConcrete3D
	# Found 1 Geometries
	#    [(1, 3, 0)] <PyMpc.MpcMeshDomain object at 0x00000297ADC03558> (# Elements: 40)
	# Found a total of 40 elements
	parameter -1001; # parameter for dTime
	parameter -1002; # parameter for dTimeCommit
	parameter -1003; # parameter for dTimeInitial
	foreach ele_id [list \\\n\t""")

	for idx, ele in enumerate(eleTags):
		f.write(f"{ele} ")
		if idx == len(eleTags)-1:
			f.write("] {\n\t")
		elif (idx+1)%20 == 0:
			f.write(" \\\n\t")

	f.write("""addToParameter -1001 element $ele_id dTime
	addToParameter -1002 element $ele_id dTimeCommit
	addToParameter -1003 element $ele_id dTimeInitial
	}
	#
	# Time-Increment Utility Functions.
	# Define a function to be called before the current time step
	proc STKO_DT_UTIL_OnBeforeAnalyze {} {
		global STKO_VAR_increment
		global STKO_VAR_time_increment
		# update the initial time and the committed time for the first time
		if {$STKO_VAR_increment == 1} {
			updateParameter -1002 $STKO_VAR_time_increment; # dTimeCommit
			updateParameter -1003 $STKO_VAR_time_increment; # dTimeInitial
		}
		# always update the current time increment
		updateParameter -1001 $STKO_VAR_time_increment; # dTime
	}
	# add it to the list of functions
	lappend STKO_VAR_OnBeforeAnalyze_CustomFunctions STKO_DT_UTIL_OnBeforeAnalyze


	recorder mpco "recorderMechanical.mpco" \\
	-N "displacement" "reactionForce" \\
	-E "material.stress" "material.strain" "material.damage" "material.cw"

	""")

	f.write("# setParameter (initStrainVol)\n")
	for idx, ε in enumerate(initialStrains):
		eleTag = eleTags[idx]
		f.write(f"setParameter -val {ε[0]} -ele {eleTag} material {0} initStrainVol\n")
		f.write(f"setParameter -val {ε[1]} -ele {eleTag} material {1} initStrainVol\n")
		f.write(f"setParameter -val {ε[2]} -ele {eleTag} material {2} initStrainVol\n")
		f.write(f"setParameter -val {ε[3]} -ele {eleTag} material {3} initStrainVol\n")

	f.write("\n# Constraints fix\n")
	f.write(constraints(xyz))

	f.write("""
	# analyses command
	domainChange
	constraints Transformation
	numberer Plain
	system UmfPack
	test NormDispIncr 1e-06 20  
	algorithm KrylovNewton -maxDim 100
	integrator LoadControl 0.0
	analysis Static
	# ======================================================================================
	# ADAPTIVE LOAD CONTROL ANALYSIS
	# ======================================================================================

	# ======================================================================================
	# USER INPUT DATA 
	# ======================================================================================

	# duration and initial time step
	set total_duration 1.0
	""")

	f.write(f"set initial_num_incr {nsteps}\n")
	

	f.write("""
	# parameters for adaptive time step
	set max_factor 1.0
	set min_factor 1e-06
	set max_factor_increment 1.1
	set min_factor_increment 1e-06
	set max_iter 20
	set desired_iter 10

	set STKO_VAR_increment 1
	set factor 1.0
	set old_factor $factor
	set STKO_VAR_time 0.0
	set initial_time_increment [expr $total_duration / $initial_num_incr]
	set STKO_VAR_initial_time_increment $initial_time_increment
	set time_tolerance [expr abs($initial_time_increment) * 1.0e-8]

	while 1 {
		
		# check end of analysis
		if {[expr abs($STKO_VAR_time)] >= [expr abs($total_duration)]} {
			if {$STKO_VAR_process_id == 0} {
				puts "Target time has been reached. Current time = $STKO_VAR_time"
				puts "SUCCESS."
			}
			break
		}
		
		# compute new adapted time increment
		set STKO_VAR_time_increment [expr $initial_time_increment * $factor]
		if {[expr abs($STKO_VAR_time + $STKO_VAR_time_increment)] > [expr abs($total_duration) - $time_tolerance]} {
			set STKO_VAR_time_increment [expr $total_duration - $STKO_VAR_time]
		}
		
		# update integrator
		integrator LoadControl $STKO_VAR_time_increment 
		
		# before analyze
		STKO_CALL_OnBeforeAnalyze
		
		# perform this step
		set STKO_VAR_analyze_done [analyze 1]
		
		# update common variables
		if {$STKO_VAR_analyze_done == 0} {
			set STKO_VAR_num_iter [testIter]
			set STKO_VAR_time [expr $STKO_VAR_time + $STKO_VAR_time_increment]
			set STKO_VAR_percentage [expr $STKO_VAR_time/$total_duration]
			set norms [testNorms]
			if {$STKO_VAR_num_iter > 0} {set STKO_VAR_error_norm [lindex $norms [expr $STKO_VAR_num_iter-1]]} else {set STKO_VAR_error_norm 0.0}
		}
		
		# after analyze
		set STKO_VAR_afterAnalyze_done 0
		STKO_CALL_OnAfterAnalyze
		
		# check convergence
		if {$STKO_VAR_analyze_done == 0} {
			
			# print statistics
			if {$STKO_VAR_process_id == 0} {
				puts [format "Increment: %6d | Iterations: %4d | Norm: %8.3e | Progress: %7.3f %%" $STKO_VAR_increment $STKO_VAR_num_iter  $STKO_VAR_error_norm [expr $STKO_VAR_percentage*100.0]]
			}
			
			# update adaptive factor
			set factor_increment [expr min($max_factor_increment, [expr double($desired_iter) / double($STKO_VAR_num_iter)])]
			
			# check STKO_VAR_afterAnalyze_done. Simulate a reduction similar to non-convergence
			if {$STKO_VAR_afterAnalyze_done != 0} {
				set factor_increment [expr max($min_factor_increment, [expr double($desired_iter) / double($max_iter)])]
				if {$STKO_VAR_process_id == 0} {
					puts "Reducing increment factor due to custom error controls. Factor = $factor"
				}
			}
			
			set factor [expr $factor * $factor_increment]
			if {$factor > $max_factor} {
				set factor $max_factor
			}
			if {$STKO_VAR_process_id == 0} {
				if {$factor > $old_factor} {
					puts "Increasing increment factor due to faster convergence. Factor = $factor"
				}
			}
			set old_factor $factor
			
			# increment time step
			incr STKO_VAR_increment
			
		} else {
			
			# update adaptive factor
			set STKO_VAR_num_iter $max_iter
			set factor_increment [expr max($min_factor_increment, [expr double($desired_iter) / double($STKO_VAR_num_iter)])]
			set factor [expr $factor * $factor_increment]
			if {$STKO_VAR_process_id == 0} {
				puts "Reducing increment factor due to non convergence. Factor = $factor"
			}
			if {$factor < $min_factor} {
				if {$STKO_VAR_process_id == 0} {
					puts "ERROR: current factor is less then the minimum allowed ($factor < $min_factor)"
					puts "Giving up"
				}
				error "ERROR: the analysis did not converge"
			}
		}
		
	}

	wipeAnalysis

	# Done!
	puts "ANALYSIS SUCCESSFULLY FINISHED"
		""")

	f.close()
	return 0

def recorder(mechPath, thermalPath, runMechanical):
	f = open(f"{thermalPath}/recorder.mpco.cdata", "r")
	data = []

	for line in f:
		if "thermal" in line:
			line = line.replace("Thermal10nt", "TenNodeTetrahedron")
			line = line.replace("None", "ASDConcrete3D")
			words = line.split(" ")
			words[6] = 1
			words[7] = 13
			words[10] = 18
			line = " ".join(map(str,words))
		data.append(line)
	f.close()

	f = open(f"{mechPath}/recorderMechanical.mpco.cdata", "w")
	for i in data:
		f.write(i)
	f.close()

	if runMechanical:
		os.system(f"OpenSees > source {mechPath}\\mainMechanical.tcl")
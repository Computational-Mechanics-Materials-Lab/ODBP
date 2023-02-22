# -*- coding: utf-8 -*-
"""
Written by J. Logan Betts for CMML 
Abaqus Python (2.7) 
Extracts a user defined node set, all_nodes, part_nodes, 
sub_nodes for avg, min, max, time. The user node set 
extracts as one 2D array to be split in the corresponding 
Python 3 script: 

"""

from odbAccess import openOdb
import numpy as np
import os

# INPUT ODB NAME: 
odbname = 'thermal_5_layer_0dt'
# INPUT THE TOTAL NUMBER OF STEPS: 
numberofsteps = 6
path = './'
myodbpath = path + odbname + '.odb'
odb = openOdb(myodbpath, readOnly=True)
# Write to new folder but check first 
savepath = './'+ odbname 
 
if not os.path.exists(savepath):
	os.makedirs(savepath)

# User defined nodeset to extract: 
newSetNodes = (568,12004,12000,11996,11992,11988, 11984, 11980, 11976, 11972, 11968)
# NODES: LB1, LB2, LB3, LB4, R1, R2, R3, R4, Y1, Y2, Y3, Y4, DB1, DB2, DB3, DB4, P1, P2, P3, P4, P5
odb.rootAssembly.NodeSetFromNodeLabels(name='myNewSet',nodeLabels=(('PART-1-1',newSetNodes),))

# Initializing iteration variable
j=0
i=0

#initalize for part 
part_max_temp = [] 
part_avg_temp = []
part_min_temp = []

#initalize for cut1 
cut1_max_temp = [] 
cut1_avg_temp = []
cut1_min_temp = []

#initalize for cut2 
cut2_max_temp = [] 
cut2_avg_temp = []
cut2_min_temp = []

#initalize for cut3 
cut3_max_temp = [] 
cut3_avg_temp = []
cut3_min_temp = []

#initalize for start 
start_max_temp = [] 
start_avg_temp = []
start_min_temp = []

#initalize for end 
end_max_temp = [] 
end_avg_temp = []
end_min_temp = []

# initalize for time & user node set
node_temp = []
time_temp = []

for x in range(numberofsteps):
	step_name= odb.steps.values()[i]
	inc = 0
	## Building in time output 
	t1 = step_name.totalTime # time till the 'Step-1'
	#Creating a for loop to iterate through all frames in the .odb
	print(odb.rootAssembly.instances["PART-1-1"].nodeSets.__dict__.keys())
	part_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['PART_NODES']
	cut1_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['CUT_1_NODES']
	cut2_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['CUT_2_NODES']
	cut3_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['CUT_3_NODES']
	start_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['START_NODES']
	end_nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['END_NODES']
	nodes = odb.rootAssembly.nodeSets['myNewSet']
	
	for x in odb.steps[step_name.name].frames:
		odbSelectResults = x.fieldOutputs['NT11']
		part_NT11 = odbSelectResults.getSubset(region=part_nodes)
		cut1_NT11 = odbSelectResults.getSubset(region=cut1_nodes)
		cut2_NT11 = odbSelectResults.getSubset(region=cut2_nodes)
		cut3_NT11 = odbSelectResults.getSubset(region=cut3_nodes)
		start_NT11 = odbSelectResults.getSubset(region=start_nodes)
		end_NT11 = odbSelectResults.getSubset(region=end_nodes)
		nodes_NT11 = odbSelectResults.getSubset(region=nodes)
		# get a numpy array
		# Not using np.copy here may work also
		p_temp = np.copy(part_NT11.bulkDataBlocks[0].data)
		c1_temp = np.copy(cut1_NT11.bulkDataBlocks[0].data)
		c2_temp = np.copy(cut2_NT11.bulkDataBlocks[0].data)
		c3_temp = np.copy(cut3_NT11.bulkDataBlocks[0].data)
		st_temp = np.copy(start_NT11.bulkDataBlocks[0].data)
		en_temp = np.copy(end_NT11.bulkDataBlocks[0].data)
		n_temp = np.copy(nodes_NT11.bulkDataBlocks[0].data)
		#convert to float64
		p_temp = p_temp.astype('float64')
		c1_temp = c1_temp.astype('float64')
		c2_temp = c2_temp.astype('float64')
		c3_temp = c3_temp.astype('float64')
		st_temp = st_temp.astype('float64')
		en_temp = en_temp.astype('float64')
		n_temp = n_temp.astype('float64')
		# convert NAN
		p_temp[p_temp == 300] = np.NAN
		c1_temp[c1_temp == 300] = np.NAN
		c2_temp[c2_temp == 300] = np.NAN
		c3_temp[c3_temp == 300] = np.NAN
		st_temp[st_temp == 300] = np.NAN
		en_temp[en_temp == 300] = np.NAN
		n_temp[n_temp == 300] = np.NAN
		# convert NAN
		p_temp[p_temp == 0] = np.NAN
		c1_temp[c1_temp == 0] = np.NAN
		c2_temp[c2_temp == 0] = np.NAN
		c3_temp[c3_temp == 0] = np.NAN
		st_temp[st_temp == 0] = np.NAN
		en_temp[en_temp == 0] = np.NAN
#		n_temp[n_temp == 0] = np.NAN
		# remove any nan's before eval
		p_temp2 = p_temp[~np.isnan(p_temp)]
		c1_temp2 = c1_temp[~np.isnan(c1_temp)]
		c2_temp2 = c2_temp[~np.isnan(c2_temp)]
		c3_temp2 = c3_temp[~np.isnan(c3_temp)]
		st_temp2 = st_temp[~np.isnan(st_temp)]
		en_temp2 = en_temp[~np.isnan(en_temp)]
		n_temp2 = n_temp[~np.isnan(n_temp)]
		if j == 0 and i == 0:
		  # part	
		  part_max_temp.append(300)
		  part_avg_temp.append(300)
		  part_min_temp.append(300)
		  # cut 1
		  cut1_max_temp.append(300)
		  cut1_avg_temp.append(300)
		  cut1_min_temp.append(300)
		  # cut 2
		  cut2_max_temp.append(300)
		  cut2_avg_temp.append(300)
		  cut2_min_temp.append(300)
		  # cut 3
		  cut3_max_temp.append(300)
		  cut3_avg_temp.append(300)
		  cut3_min_temp.append(300)
		  #start
		  start_max_temp.append(300)
		  start_avg_temp.append(300)
		  start_min_temp.append(300)	
		  # end
		  end_max_temp.append(300)
		  end_min_temp.append(300)	
		  end_avg_temp.append(300)
		  # node
		  node_temp.append(n_temp)
		else: 	 
		  # part	
		  part_max_temp.append(np.max(p_temp2) if len(p_temp2) > 0 else np.nan)
		  part_avg_temp.append(np.mean(p_temp2) if len(p_temp2) > 0 else np.nan)
		  part_min_temp.append(np.min(p_temp2) if len(p_temp2) > 0 else np.nan)
		  # cut 1
		  cut1_max_temp.append(np.max(c1_temp2) if len(c1_temp2) > 0 else np.nan)
		  cut1_avg_temp.append(np.mean(c1_temp2) if len(c1_temp2) > 0 else np.nan)
		  cut1_min_temp.append(np.min(c1_temp2) if len(c1_temp2) > 0 else np.nan)
		  # cut 2
		  cut2_max_temp.append(np.max(c2_temp2) if len(c2_temp2) > 0 else np.nan)
		  cut2_avg_temp.append(np.mean(c2_temp2) if len(c2_temp2) > 0 else np.nan)
		  cut2_min_temp.append(np.min(c2_temp2) if len(c2_temp2) > 0 else np.nan)
		  # cut 3
		  cut3_max_temp.append(np.max(c3_temp2) if len(c3_temp2) > 0 else np.nan)
		  cut3_avg_temp.append(np.mean(c3_temp2) if len(c3_temp2) > 0 else np.nan)
		  cut3_min_temp.append(np.min(c3_temp2) if len(c3_temp2) > 0 else np.nan)
		  #start
		  start_max_temp.append(np.max(st_temp2) if len(st_temp2) > 0 else np.nan)
		  start_avg_temp.append(np.mean(st_temp2) if len(st_temp2) > 0 else np.nan)
		  start_min_temp.append(np.min(st_temp2) if len(st_temp2) > 0 else np.nan)	
		  # end
		  end_max_temp.append(np.max(en_temp2) if len(en_temp2) > 0 else np.nan)
		  end_avg_temp.append(np.mean(en_temp2) if len(en_temp2) > 0 else np.nan)
		  end_min_temp.append(np.min(en_temp2) if len(en_temp2) > 0 else np.nan)
		  # node
		  node_temp.append(n_temp)
		#Progress report and save output was numpy array
		print('\nExtracting from Frame:\t'+str(j))
# convert temp 1s to list
		n_temp = n_temp.tolist()
		p_temp = p_temp.tolist()
		c1_temp = c1_temp.tolist()
		c2_temp = c2_temp.tolist()
		c3_temp = c3_temp.tolist()
		st_temp = st_temp.tolist()
		en_temp = en_temp.tolist()
# convert temp 2s to list
		p_temp2 = p_temp2.tolist()
		c1_temp2 = c1_temp2.tolist()
		c2_temp2 = c2_temp2.tolist()
		c3_temp2 = c3_temp2.tolist()
		st_temp2 = st_temp2.tolist()
		en_temp2 = en_temp2.tolist()
		# Time output test
		tinst = x.frameValue		# time for the frame in the current step
		tcurrent = t1 + tinst       # toal current time
 		time_temp.append(tcurrent) # write out to a 1,1 array
		j += 1
	i += 1

# Change Directory and write: 
os.chdir(savepath)

# write out part
np.save('part_avg_NT11_'+odbname+'.npy', part_avg_temp)
np.save('part_max_NT11_'+odbname+'.npy', part_max_temp)
np.save('part_min_NT11_'+odbname+'.npy', part_min_temp)

# write out cut1
np.save('cut1_avg_NT11_'+odbname+'.npy', cut1_avg_temp)
np.save('cut1_max_NT11_'+odbname+'.npy', cut1_max_temp)
np.save('cut1_min_NT11_'+odbname+'.npy', cut1_min_temp)

# write out cut2
np.save('cut2_avg_NT11_'+odbname+'.npy', cut2_avg_temp)
np.save('cut2_max_NT11_'+odbname+'.npy', cut2_max_temp)
np.save('cut2_min_NT11_'+odbname+'.npy', cut2_min_temp)

# write out cut3
np.save('cut3_avg_NT11_'+odbname+'.npy', cut3_avg_temp)
np.save('cut3_max_NT11_'+odbname+'.npy', cut3_max_temp)
np.save('cut3_min_NT11_'+odbname+'.npy', cut3_min_temp)

# write out start nodes
np.save('start_avg_NT11_'+odbname+'.npy', start_avg_temp)
np.save('start_max_NT11_'+odbname+'.npy', start_max_temp)
np.save('start_min_NT11_'+odbname+'.npy', start_min_temp)

# write out end nodes
np.save('end_avg_NT11_'+odbname+'.npy', end_avg_temp)
np.save('end_max_NT11_'+odbname+'.npy', end_max_temp)
np.save('end_min_NT11_'+odbname+'.npy', end_min_temp)

# write out user array & time & usr node info 
np.save('time'+odbname+'.npy', time_temp)		
np.save('nodes_NT11_'+odbname+'.npy', node_temp)

odb.close()

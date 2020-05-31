data_file = open('/root/mlops/data.txt','r')
input_file = open('/root/mlops/input.txt','r')
accuracy_file = open('/root/mlops/accuracy.txt','r')

data = data_file.read()
data = data.split('\n')

old_accuracy = float(data[0])		  	  #read old accuracy stored at top of line
layer = int(data[1])				      #implies layer 1 for convolve and 2 for fully connected
line = int(data[2])					      #implies line number which has the data to upon which the changes were done and will be further done too
cp_line = line % 3						  #in case of convolve layer (line % 3) giving output of lines in terms of 1 for no of filters , 2 for strides and 0 for pools
entered_data = int(data[3])				  #implies the changed data which was taken by program as input
old_data = int(data[4])					  #this is the value from which changes in the entered_data will begin for the first time in the layer			
index_fc = int(data[5])					  #line number of the input which shall specify the number of fully connected layers in the program

new_accuracy = float(accuracy_file.read()) 			  #read the new accuraacy achieved

#Note : For the first time old data and entered data will be same

inputs = input_file.read()
inputs = inputs.split('\n')

'''You must be thinking whats the use of this program seeing the accuracy when its called only for increasing the accuracy...
	Well , this is because thie original program has multiple input parameters whose value can be increased or decreased and we cannot increase or decrease the value of only one input for a greater accuracy.
	So , this program will do the accuracy check to decide which parameter's value to be changed'''

if new_accuracy > old_accuracy and new_accuracy - old_accuracy >= 0.00001 :
	'''The data we modified helped increase accuracy and can be further increased
		An increase in accuracy greater than or equal to 0.001% is atleast required to consider it as helping in giving a beter output
		The above value (0.001%) is high considering the gpu capacity of my system, if the gpu capacity of your system is good its recommended to further decrease its value to less than or equal to 0.0000001% to gain more better results'''
	old_accuracy = new_accuracy
	if layer == 1:
		if cp_line == 1:
			entered_data = entered_data * 2
		else :
			entered_data += 1			  #leaving the number of filters in convolve layer , rest all data in the convolve layer namely strides and pool size to be incrmented by 1 only
	else :
		entered_data += 100				  #the dense layer neurons to be incremented by 100				
	inputs[line] = str(entered_data)	  #updating the input file with the changes							

else:
	#The data we modified reduced frequency and so we want to make amendments
	if layer == 1 :	#convolve layer
		if cp_line == 1 :	#here number of filters are input
			if entered_data//2 == old_data : #This condition being true implies that the addition of an extra convolution layer was not that too fruitful and so its better to remove the extra convolution layer
				'''Here 2 steps will be taken :
						Step 1 : Remove the data added for the convolve layer 
						Step 2 : Add the data for the fully connected fc layer'''
				'''Implementing Step 1 :
					The data of convolve layer takes 3 lines of inputs which are number of filters , size of filters and size of pooling respectively.
					Thus removing this convolve layer means removing all these 3 lines from input.
					The data of input.txt file is stored in inputs in form of list containing strings of data.
					We know the line number from the variable line where the first input of the conolve layer to be removed is stored and so modifmodifying inputs to store data only till before it.'''
				inputs = inputs[0:line]
				#Implementing Step 2
				inputs.append('1') 						#Now there shall be one fully connected layer
				layer = 2								#for now all the modifications will be done in the fully conncted layer
				index_fc = line    						#for this shall hold the index of the input specifying number of fc layers
				inputs.append('100')					#This shall be the initial value for the dense layer neurons
				old_data = 100							#for this shall be the initial data of this fully connected layer
				entered_data = 100						#for this will be the data which user shall input in the program
				line = line + 1					 	    #for this shall point to the line whose data is changed
				inputs[0] = str(int(inputs[0]) - 1)		#decreasing the number of convolve layers in the input 
			else :
				'''Coming here means that this convolve layer is needed but the number of filters is too much and needs to be decreased to its previous value.
					This also means that now its time to shift to the next layer of data input i.e. the stride size'''
				inputs[line] = str(entered_data//2)		#modified the data to its previous value
				line = line + 1							#shifted to the next input line#for on this data the calculations will be made
				entered_data = 3						#for on this data the calculations will be made in filter size
				old_data = 2							#he beginning value of the filter size
				inputs[line] = str(entered_data)		#for now the input file will have this new data for the filter size
		elif cp_line == 2:
			#coming here changes need to be reveresed in the filter size and shift to pool size changes
			inputs[line] = str(entered_data - 1)		#modified the data to its previous value
			line = line + 1								#shifted to the next input line
			entered_data = 3							#for on this data the calculations will be made in pool size
			old_data = 2								#he beginning value of the pool size
			inputs[line] = str(entered_data)			#for now the input file will have this new data for the pool size		
		elif cp_line == 0:
			'''Coming here marks the end of this convolve layer and we now we want o add another convolve layer
				But first lets shift the pool size to its previous value which gave us an increased accuracy.'''
			inputs[line] = str(entered_data - 1)		#modified the data to its previous value
			line = line + 1								#shifted to the next input line
			old_data = int(inputs[line - 3])			#nebacw convolve layer will have filters double the number that wew present in the previous convolve layer ( the previous layer data is present 3 lines back )
			entered_data = old_data * 2					#for on this data the calculations will be made in the number of filters
			inputs[0] = str(int(inputs[0]) + 1) 		#increasing the number of convolve layers in the input
			inputs = inputs[0:line]						#getting in the inputs the data of the layers we have till now
			inputs.append(str(entered_data))			#inserting the new initial inputs for the new convolve layer , inserting number of filters
			inputs.append('2')							#inserting filter size
			inputs.append('2')							#inserting pool size
			inputs.append('0')							#for presently there is no intermediate fc layer so number will be 0 , the last data of input file stores number of intermediate fc layers unless we finally add an fc layer
			index_fc = line + 3							#for the index of the fc layer number shall be present at this line number only
	else:
		'''Coming here means now we have to add another fully connecte layer
			Tasks to do : 
			1)Update the index_fc for addition of another fc layer
			2)Decrease the number of the fc layer neurons to its previous one ie. 100 less from the entered data
			3)begin another fc layer with neurons equal to that of previous layer'''
		noOfLayers = int(inputs[index_fc])+1		#increasing the number of fc layers
		inputs[index_fc]=str(noOfLayers)			#making the changes in the input file as number of fc lyers increased
		entered_data -= 100							#reducing the number of neurons in the previous layer
		old_data = entered_data						#since this sahll be the base or initial value for this layer
		inputs[line] = str(entered_data)			#decreasing the number of neurons in the previous layer in the input file
		line += 1									#since a new fc layer added to point at
		inputs.append(str(entered_data))			#append the new fc layer data in the input file

#closing both the files opened in read mode
data_file.close()
input_file.close()

#opening both the files in write mode
data_file = open('/home/jyotirmaya/ws/mlops1/data.txt','w')
input_file = open('/root/mlops/input.txt','w')

data_file_data = str(old_accuracy) + '\n' + str(layer) + '\n' + str(line) + '\n' + str(entered_data) + '\n' + str(old_data) + '\n' + str(index_fc)

data_file.write(data_file_data)							#The data.txt file ready to help in making changes in the next call of tewaker.py

data_file.close()

input_file_data = '\n'.join(inputs)

input_file.write(input_file_data)						#The input.txt file ready to provide inputs to the program kerasCNN.py program.

input_file.close()


'''
Below are my learnings gathered from some hit and trials on the file operations and I used this learning for writing the complete tweaker.py program.

My Learnings : 

For reading :-
The readLine function captures till line end including the '\n'.
The last line does mot have '\n'.
To get the end of file in Python , we need to use a trick that data received on reading will give ''.

For writing :-
The write function does not include '\n' in the end on it own.
The write function takes only string data to write in file.

For converting string with spaces intlo list of words eg : 'Hello World' to ['Hello','World'] :-
x = x.split()			(x is my string)
Place in split paranthesis on what basis you want to split like split(',') for comma .By default its for space.

For converting a list of strings only into a complete string with the delimeter of my choice eg. ['a','b','c'] to 'a b c'
x = ' '.join(x)			(x is my list)
Replace ' ' with the delimeter of your choice like ',' for comma.
 '''

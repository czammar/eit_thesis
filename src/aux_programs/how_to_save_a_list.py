# Example list
thelist=  [11, 12, 13]

# Open the file for writing
dataFile = open('writetest.txt', 'w')

# Loop through each item in the list
# and write it to the output file.
for eachitem in thelist:
    dataFile.write( str(eachitem) + '\n')

# Close the output file
dataFile.close()

# Also  you can use

np.save(outfile,thelist) # to save
#np.load('outfile') # to open

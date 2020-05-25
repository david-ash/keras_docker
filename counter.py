file = open('variables_file', 'r')
line = file.readlines()
number=line[0][6:len(line[0])-1]
number=int(number)+1
number=str(number)
line[0] = line[0][0:6] + number + line[0][len(line[0])-1]
file = open('variables_file', 'w')
file.writelines(line)
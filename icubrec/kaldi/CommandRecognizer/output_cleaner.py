import sys
import re

def output_cleaner(file_path):
    f = open(file_path, "r")
    file_line = f.readlines()
    f.close()
    line_to_save = "ERROR"
    for line in file_line:
        if line[0].isdigit() and line[8].isupper():
            line_to_save = line[8:]
    f = open(file_path, "w")
    f.write(line_to_save)
    f.close()

def output_cleaner_dnn(file_path, latgen_file_path, threshold=1.5, first_char_pos=8):
    # Checking output of latgen for warnings
    f = open(latgen_file_path, "r")
    file_line = f.readlines()
    f.close()
    line_to_save = ''
    for line in file_line:
	if line.startswith('WARNING'):
            print('WARNING: partial recognition')
	    return False
    # Extracting the recognized command
    f = open(file_path, "r")
    file_line = f.readlines()
    f.close()
    line_to_save = ''
    cost = None
    for line in file_line:
        if line[0].isdigit() and line[first_char_pos].isupper():
            line_to_save = line[first_char_pos:]
        res = re.search('([0-9\.]+)\s\[acoustic\]', line)
        if res is not None:
            cost = float(res.group(1))
    if cost is None:
        print('WARNING: no cost found')
        return False
    print('COST:', cost)
    if line_to_save.startswith('DISCARD'):
        print('WARNING: SPN or garbage detected')
        return False
    if cost > threshold:
        print('WARNING: sentence discard, cost above threshold')
	return False
    if line_to_save == '':
        print('WARNING: transcription empty or not found')
        return False

    #line_to_save += str(cost) + '\n'
    f = open(file_path, "w")
    f.write(line_to_save)
    f.close()
    return True

if __name__ == '__main__':
    file_path= sys.argv[1]
    output_cleaner(file_path)

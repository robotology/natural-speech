import yarp
def write_on_output_port(cmd_file,yarp_port=None):
    with open(cmd_file) as fh:
        #word_list = [l.strip('\n').split(' ') for l in fh.readlines()]
        #for word in word_list[0]:#we have only one line for each file
        #    if word != '':
        #        if word != "ERROR":
        #            yarp_port.write(yarp.Value(word))
        #        else:
        #            pass
        
        # TODO: create a bottle with a list of values
        lines = [l.rstrip('\n') for l in fh.readlines()]
        bout = yarp.Bottle()
        for l in lines:
	    for w in l.split():
	        bout.addString(w)
        yarp_port.write(bout)
if __name__ == '__main__':
    write_on_output_port("../pipeline/save/537_842.txt")

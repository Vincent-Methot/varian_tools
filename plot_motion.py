import numpy as np
import matplotlib.pyplot as plt
import sys



fichier = open(sys.argv[1])
motion = fichier.read()
motion = motion.split('\n')
motion = motion[:-1]
motion = [m.split() for m in motion]
motion = np.array(motion, 'float32')


for i in range(1, len(motion[0,:])):
    plt.plot(np.arange(len(motion[:,i])), motion[:,i])

plt.show()
# Program to plot the variation of my height as the years pass by

from matplotlib import pyplot as plt
from itertools import compress
years = list(compress(range(2000,2019), [1]*18))
height = [50, 60, 70 ,80, 90, 100, 110 ,120, 130 , 135 , 138, 140, 144, 150, 160, 170, 172, 173]
plt.plot(years, height, color='green', linestyle='solid', marker='o')
plt.title("Variation of Height with age")
plt.ylabel("in cms")
plt.show()


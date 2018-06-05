from matplotlib import pyplot as plt


# Dataset 


grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade:grade // 100
histogram = Counter(decile(grade) for grade in grades)

# Plot Code

plt.bar([x-4 for x in histogram.keys()], histogram.value(), 8)
plt.title("Histogram for grades --")
plt.axis([-5,105,0,5])
plt.ylabel("Marks scored out of 100 ")
plt.show()
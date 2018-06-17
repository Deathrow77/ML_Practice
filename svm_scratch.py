
import matplotlib.pyplot as plt
import numpy as np

data_dict = {-1:np.array([[1,7],[2,8],[3,8],]), 1:np.array([[5,1],[6,-1],[7,3],])}

# SVM Class

class SupportVectorMachine:
    def __init__(self, visualization=True): 
        self.visualization = visualization
        self.colors = {-1:'r', 1:'b'}

        # Create a condition if visualization == True
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    def fit(self, data):
        # Linking data argument to self
        self.data = data
            # Creating our results dictionary
            # Format {||w|| : [w,b]}
        opt_dict = {}

        # Create a list just to calculate the min and max feature values
        all_data = []
        #Define transforms to be applied on w
        transforms = [[1,1], [-1,1],[-1,-1], [1,-1]]

        # iterate over the data and append every feature to the temporary list
        for yi in self.data:
           for features in self.data[yi]:
               for feature in features:
                   all_data.append(feature)

        # Extract max and min feature values and nullify the temporary list
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        all_data = None
        # Define the stepSizes, the multipliers, and the latest_optimum value which will be our starting point for calculation
        step_sizes = [self.max_feature_value*0.1, self.max_feature_value*0.01, self.max_feature_value*0.001]
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        # For all step values perform the calculations mentioned below
        for step in step_sizes:

            # create a w array which will store the latest optimum values
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange((-1*(self.max_feature_value)*b_range_multiple), (self.max_feature_value)*b_range_multiple, step*b_multiple):
                    for transformation in transforms:
                        # Apply transformations on w and create a flag to check for the constraint of Suppor Vector
                        w_t = w*transformation
                        found_option = True
                        # Weakest Link : Iterating over all the data and checking for the SVM Constraint
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not ((yi*((np.dot(w_t, xi))+b))>=1): # Check if the constrainst is not met
                                    found_option = False 
                        # if flag is true, append it to the opt_dict { ||w|| : [w,b]}
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                # If the ||w|| < 0 then the value is optimized, else keep reducing step value from w
                if w[0]<0:
                    optimized = True
                else:
                    w=w-step
        # Sort the opt_dict on the basis of the value of ||w||        
        norms = sorted([n for n in opt_dict])
        opt_choice = opt_dict[norms[0]] # Return the value for the key : value pair which has the lowest key value
        # ||W|| : [w,b]
        self.w = opt_choice[0]
        self.b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step*2

        # Printing the results 

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, " : ", yi*(np.dot(xi, self.w) + self.b))


    def predict(self, features):
        # the class will be predicted by sign(x.w + b)
        classification =  np.sign(np.dot(features, self.w) + self.b)
        # If visualization is true and classification != 0, we will scatter plot the point
        if classification!=0 & self.visualization:
            self.ax.scatter(features[0], features[1], s=50, marker='o', c=self.colors[classification])
        # Return the prediction
        return classification


    def visualize(self):
        # Scatter plot all the data points
        [[self.ax.scatter(x[0], x[1], s=50, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        # Define a hyperplane equation
        def hyperplane(x,w,b,v):
            return ((-w[0]*x - b + v)/w[1])
        # Confine hyperplane to a certain limits
        data_range = (self.min_feature_value*0.9, self.min_feature_value*1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]
        # Visualisation of the +ve Support Vectors (v=1)
        psv1 = hyperplane(hyp_x_min, self.w, self. b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # Visualisation of the -ve Support Vectors (v=-1)

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # Visualisation of the Decision Boundary (v=0)

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


SupVM = SupportVectorMachine()
SupVM.fit(data=data_dict)
SupVM.visualize()
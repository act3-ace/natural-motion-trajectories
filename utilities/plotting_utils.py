import matplotlib as mpl 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d


class PlottingFcns:

    # Style plot 
    marker_size = 1.5
    line_width = 1.25
    fig = plt.figure(figsize=(20,5))
    axis_font = 9
    ax_label_font = 11
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : axis_font}
    mpl.rc('font', **font)

    # -------------------------------
    # Plot 2D Scenario
    # -------------------------------
    @classmethod 
    def plot2D(cls, plot_ops_flags, scenario, name):

        ax1 = plt
        ax1.grid()

        if plot_ops_flags["Truth"]:
            X = scenario["truthResponse"]
            ax1.plot(X[0,:],X[1,:], color='coral', markersize=cls.marker_size, alpha=0.8,label='Truth')
            ax1.plot(X[0,0],X[1,0],'kx')

        if plot_ops_flags["Measured"]:
            X_meas = scenario["measuredResponse"]
            ax1.plot(X_meas[0,:],X_meas[1,:], color='green', markersize=cls.marker_size, alpha=0.6,label='Measured')

        if plot_ops_flags["Estimated"]:
            X_hat = scenario["stateEstimation"]
            ax1.plot(X_hat[0,:],X_hat[1,:], color='blue', markersize=cls.marker_size, alpha=0.4,label='Estimated')

        
        ax1.xlabel("$x-position$", fontsize=cls.ax_label_font)
        ax1.ylabel("$y-position$", fontsize=cls.ax_label_font)
        plt.title("In-Plane Trajectory", fontsize=cls.ax_label_font)
        ax1.legend()

        # Save and Show 
        plt.savefig(name+"_2D_plot")
        plt.show()

    # -------------------------------
    # Plot 3D Scenario
    # -------------------------------
    @classmethod 
    def plot3D(cls, plot_ops_flags, scenario, name):

        fig = plt.figure()
        ax1 = plt.axes(projection='3d')

        plt.show()

        if plot_ops_flags["Truth"]:
            X = scenario["truthResponse"]
            ax1.plot3d(X[0,:],X[1,:], X[2,:],color='coral', markersize=cls.marker_size, alpha=0.8,label='Truth')
            ax1.plot3d(X[0,0],X[1,0],X[2,0],'kx')

        if plot_ops_flags["Measured"]:
            X_meas = scenario["measuredResponse"]
            ax1.plot(X_meas[0,:],X_meas[1,:], color='green', markersize=cls.marker_size, alpha=0.6,label='Measured')

        if plot_ops_flags["Estimated"]:
            X_hat = scenario["stateEstimation"]
            ax1.plot(X_hat[0,:],X_hat[1,:], color='blue', markersize=cls.marker_size, alpha=0.4,label='Estimated')

        
        ax1.xlabel("$x-position$", fontsize=cls.ax_label_font)
        ax1.ylabel("$y-position$", fontsize=cls.ax_label_font)
        ax1.zlabel("$z-position$", fontsize=cls.ax_label_font)
        plt.title("3D Trajectory", fontsize=cls.ax_label_font)
        ax1.legend()

        # Save and Show 
        plt.savefig(name+"_3D_plot")
        #plt.show()

    # -------------------------------
    # Plot State Estimation Error
    # -------------------------------
    @classmethod
    def plot_state_error(cls, scenario, name):

        t = scenario["scenarioTime"]
        state_error = scenario["stateError"]

        ax2 = plt
        ax2.grid()
        ax2.plot(t,state_error[0,:],color='blue',markersize=cls.marker_size, alpha=0.8,label='$x-error$')
        ax2.plot(t,state_error[1,:],color='red',markersize=cls.marker_size, alpha=0.8,label='$y-error$')
        #ax2.set_ylim(-1,1)
        ax2.xlabel("$time$", fontsize=cls.ax_label_font)
        ax2.ylabel("$state-error$", fontsize=cls.ax_label_font)
        ax2.legend()
        
        plt.title("State-Error vs. Time", fontsize=cls.ax_label_font)

        # Save and Shot
        plt.savefig(name+"_state_error")
        #plt.show()
    
    @classmethod
    def Plotter(cls, flags, scenario, name):
        if flags["2D_trajectory"]["plot"]: cls.plot2D(flags["2D_trajectory"]["plotOptions"], scenario, name) # Plot 2D Scenario

        if flags["3D_trajectory"]["plot"]: cls.plot3D(flags["3D_trajectory"]["plotOptions"], scenario, name) # Plot 3D Scenario

        if flags["estimation_error"]: cls.plot_state_error( scenario, name) # Plot 2D Scenario

            




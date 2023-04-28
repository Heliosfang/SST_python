import math
import random
import shapely
import shapely.ops
import numpy as np


class KinematicsNoAcc:
    def __init__(self,initial_state,wheel_base,map,v_range,delta_range,min_time_steps,max_time_steps,dt=0.1) -> None:
        # define the initial state,wheel_base,dt
        self.initial_state_ = [initial_state.x,initial_state.y,initial_state.z]
        self.wheel_base = wheel_base

        # get the map
        self.map = map

        # have the random range
        self.v_min = v_range[0]
        self.v_max = v_range[1]
        self.steering_min = delta_range[0]
        self.steering_max = delta_range[1]

        # define the step time and num of steps
        self.dt = dt
        self.min_time_steps = min_time_steps
        self.max_time_steps = max_time_steps


        # define the initial forward and backward result state
        self.result = []
        self.state_vec = []
        self.state_vec.append(self.initial_state_)

        self.f_result = self.initial_state_
        self.b_result = self.initial_state_


    def forwardPropagate(self):
        """return a valid state that can be used as the next child state"""
        fu = self.forwardUpdate()
        if not fu:
            return False
        
        
        forward_result = []
        forward_result.append(shapely.points(self.state_vec).tolist())
        forward_result.append(self.dt*self.forward_steps)
        return forward_result


    def forwardUpdate(self):
        """Check whether the random state and control makes the car out of map"""
        control = []
        # add a random steering
        control.append(random.uniform(self.steering_min,self.steering_max))
        # add a random speed
        control.append(random.uniform(self.v_min,self.v_max))

        self.forward_steps = random.randint(self.min_time_steps,self.max_time_steps)


        for i in range(self.forward_steps):
            self.f_result[0] = self.f_result[0] + control[1]*math.cos(control[0]+self.f_result[2])*self.dt
            self.f_result[1] = self.f_result[1] + control[1]*math.sin(control[0]+self.f_result[2])*self.dt
            self.f_result[2] = self.f_result[2] + control[1]*math.tan(control[0])/self.wheel_base*self.dt


            if shapely.contains(self.map,shapely.Point(self.f_result[0],self.f_result[1])):
                self.f_result = self.initial_state_
                self.state_vec = []
                self.state_vec.append(self.initial_state_)
                return False
            while(self.f_result[2]<-math.pi):self.f_result[2]+=2*math.pi
            while(self.f_result[2]>math.pi):self.f_result[2]-=2*math.pi
            self.state_vec.append([self.f_result[0],self.f_result[1],self.f_result[2]])

        return True
    

    def backwardPropagate(self):
        """return a valid state that can be used as the next child state"""
        bu = self.backwardUpdate()
        if not bu:
            return False
        backward_result = []
        backward_result.append(shapely.points(self.state_vec).tolist())
        backward_result.append(self.dt*self.backward_steps)
        return backward_result


    def backwardUpdate(self):
        """Check whether the random state and control makes the car out of map"""
        control = []
        # add a random steering
        control.append(random.uniform(self.steering_min,self.steering_max))
        # add a random speed
        control.append(random.uniform(self.v_min,self.v_max))

        self.backward_steps = random.randint(self.min_time_steps,self.max_time_steps)

        for i in range(self.backward_steps):
            self.b_result[0] = self.b_result[0] - control[1]*math.cos(control[0]+self.b_result[2])*self.dt
            self.b_result[1] = self.b_result[1] - control[1]*math.sin(control[0]+self.b_result[2])*self.dt
            self.b_result[2] = self.b_result[2] - control[1]*math.tan(control[0])/self.wheel_base*self.dt


            if shapely.contains(self.map,shapely.Point(self.b_result[0],self.b_result[1])):
                self.b_result = self.initial_state_
                self.state_vec = []
                self.state_vec.append(self.initial_state_)
                return False
            while(self.b_result[2]<-math.pi):self.b_result[2]+=2*math.pi
            while(self.b_result[2]>math.pi):self.b_result[2]-=2*math.pi
            self.state_vec.append([self.b_result[0],self.b_result[1],self.b_result[2]])

        return True
    






        







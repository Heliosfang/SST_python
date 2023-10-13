import shapely
import shapely.ops
import numpy as np
from shapely.geometry import shape
import random
from rtree import index
from KinematicsNoAcc import KinematicsNoAcc
import time
import ctypes
from threading import Thread
from threading import Lock
from time import sleep
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class CircleNode:
    """A node storing the best cost and coordinate of the tree"""
    def __init__(self,id,center,active_p) -> None:
        self.id = id
        self.center = center
        self.active_n = active_p #the current active node in the rtree
        # self.active_cost = active_cost #the current best cost in the node 

class Node:
    def __init__(self,id,pose,cost,parent = None,children = [],active=True):
        self.id = id
        self.pose =pose
        self.cost = cost
        self.parent = parent
        self.children = children
        # self.path_from_parent = path_from_parent
        self.active = active
        if parent is not None:
            parent.children.append(self)


class SST:
    def __init__(self,start,goal,map,wheel_base,v_range,delta_range,max_running_time,min_running_time = 0,bi_tree=True,time_steps_range=[1,10]):
        

        # create a shared lock
        self.forward_lock = Lock()
        self.backward_lock = Lock()
        # buffer the map in a safe consideration

        buffer_map = []
        for n in map.geoms:
            m_b = shapely.Polygon(n.buffer(0.5,join_style=1))
            buffer_map.append(m_b)

        self.map = shapely.unary_union(buffer_map)

        # initial the running time
        self.maximum_time = max_running_time
        self.minimum_time = min_running_time

        # initial the time steps
        self.min_time_steps = time_steps_range[0]
        self.max_time_steps = time_steps_range[1]


        # initial the start time
        self.start_time = time.time()

        # initial the best cost
        self.forward_best_cost_ = float('Inf')

        # define the initial state of the tree
        self.start_pos = start
        self.goal_pos = goal

        # configuration of the car for the given model
        self.wheel_base = wheel_base
        self.v_min =  v_range[0]
        self.v_max = v_range[1]
        self.steering_min = delta_range[0]
        self.steering_max = delta_range[1]

        # set whether the tree constructed on both side
        self.bi_tree = bi_tree

        # initial the forward tree
        self.forward_tree = index.Index() #forward rtree
        # self.forward_tree_temp = index.Index() #backup forward rtree
        self.forward_circle_tree = index.Index()
        self.circle_radius = 0.5 #define the radius of the circle node
        self.forward_snode_id = 0
        self.forward_cnode_id = 0

        self.f_treeVector = []
        self.f_circleVector = []
        self.forward_last_node = None


        # create the first node
        f_parent0 = Node(self.forward_snode_id,self.start_pos,0,None,[])
        self.f_treeVector.append(f_parent0)
        f_circleNode0 = CircleNode(self.forward_cnode_id,f_parent0.pose,id(self.f_treeVector[self.forward_snode_id]))
        self.f_circleVector.append(f_circleNode0)
        # print(f_parent0.pose.x,f_parent0.pose.y)

        # self.f_circleVector.append(f_circleNode0)
        # self.f_treeVector.append(f_parent0)

        self.forward_tree.insert(f_parent0.id,(f_parent0.pose.x,f_parent0.pose.y),id(self.f_treeVector[self.forward_snode_id]))
        # self.forward_tree_temp.insert(f_parent0.id,(f_parent0.pose.x,f_parent0.pose.y),id(self.f_treeVector[f_parent0.id]))

        self.forward_circle_tree.insert(f_circleNode0.id,(f_circleNode0.center.x,f_circleNode0.center.y),id(self.f_circleVector[self.forward_cnode_id]))
        self.forward_last_node = self.f_treeVector[self.forward_snode_id]
        self.f_reach_end = False
        self.fsolutionvector = None



        # if bi_tree, initial the backward tree
        if(self.bi_tree):
            self.backward_tree = index.Index() #forward rtree
            # self.forward_tree_temp = index.Index() #backup forward rtree
            self.backward_circle_tree = index.Index()
            
            self.backward_snode_id = 0
            self.backward_cnode_id = 0

            self.b_treeVector = []
            self.b_circleVector = []
            self.backward_last_node = None

            self.backward_best_cost_ = float('Inf')



            # create the first node
            b_parent0 = Node(self.backward_snode_id,self.goal_pos,0,None,[])
            self.b_treeVector.append(b_parent0)
            b_circleNode0 = CircleNode(self.backward_cnode_id,b_parent0.pose,id(self.b_treeVector[self.backward_snode_id]))
            self.b_circleVector.append(b_circleNode0)
            # print(b_parent0.pose.x,b_parent0.pose.y)

            # self.f_circleVector.append(f_circleNode0)
            # self.f_treeVector.append(f_parent0)

            self.backward_tree.insert(b_parent0.id,(b_parent0.pose.x,b_parent0.pose.y),id(self.b_treeVector[b_parent0.id]))
            # self.forward_tree_temp.insert(f_parent0.id,(f_parent0.pose.x,f_parent0.pose.y),id(self.f_treeVector[f_parent0.id]))

            self.backward_circle_tree.insert(b_circleNode0.id,(b_circleNode0.center.x,b_circleNode0.center.y),id(self.b_circleVector[self.backward_cnode_id]))
            self.backward_last_node = self.b_treeVector[self.backward_snode_id]

            self.b_reach_end = False
            self.bsolutionvector = None

        # self.threshold = 0.5

        # have the boundary of the map, [min_x, min_y, max_x, max_y]
        self.bound = shapely.bounds(self.map).tolist()
        self.flag = True
        forward_thread = Thread(target=self.get_forward_tree)
        # if self.bi_tree:
            # backward_thread = Thread(target=self.get_backward_tree)

        forward_thread.start()
        # jump_start_backward = True
        # if self.bi_tree:
        #     backward_thread.start()

        time_counter = 2.0
        while time.time()-self.start_time<self.maximum_time:
            
            if time.time()-self.start_time//1 > time_counter:
                print("time runed on sst is :%f, while the minimum running time is: %f" %(time.time()-self.start_time,self.minimum_time))
                time_counter+=2
            # forward_thread.join()
            # if self.bi_tree:
            #     backward_thread.join()
            if time.time()-self.start_time>self.minimum_time:
                # if time.time()-self.start_time>5.0 and self.bi_tree and jump_start_backward:
                    # backward_thread.start()
                    # jump_start_backward = False
                if self.bi_tree:
                    if self.f_reach_end or self.b_reach_end:
                        self.flag=False
                        break
                    sleep(0.1)
                else:
                    if self.f_reach_end:
                        self.flag=False
                        break
                    sleep(0.1)
            else:
                sleep(0.1)

            # if self.f_reach_end or self.b_reach_end:
            #     break
            
            # i+=1
            # print(i)

        # print(time.time())
        self.flag = False
        forward_thread.join()
        # if self.bi_tree and not jump_start_backward:
            # backward_thread.join()
        # print("22222: ",time.time())

    def get_forward_tree_node_list(self):
        return self.f_treeVector
    
    def get_backward_tree_node_list(self):
        return self.b_treeVector
    
    def get_tree_reached(self):
        if self.bi_tree:
            return [self.f_reach_end,self.b_reach_end]
        else:
            return self.f_reach_end




    def get_forward_nearest_point(self,state):
        q = list(self.forward_tree.nearest((state[0],state[1]),1,objects=True))
        # min_node = None
        # min_cost = float('Inf')
        # debug_q = ctypes.cast(q[0].object,ctypes.py_object).value
        # for i in range(len(q)):
        ni = ctypes.cast(q[0].object, ctypes.py_object).value
            # if shapely.distance(shapely.Point(state[0],state[1]),shapely.Point(ni.pose.x,ni.pose.y))<min_cost:
                # min_cost = shapely.distance(shapely.Point(state[0],state[1]),shapely.Point(ni.pose.x,ni.pose.y))
        # min_node = ni
        
        return ni
        
    def get_backward_nearest_point(self,state):
        q = list(self.backward_tree.nearest((state[0],state[1]),1,objects=True))
        # min_node = None
        # min_cost = float('Inf')
        # debug_q = ctypes.cast(q[0].object,ctypes.py_object).value
        # for i in range(len(q)):
        ni = ctypes.cast(q[0].object, ctypes.py_object).value
            # if shapely.distance(shapely.Point(state[0],state[1]),shapely.Point(ni.pose.x,ni.pose.y))<min_cost:
                # min_cost = shapely.distance(shapely.Point(state[0],state[1]),shapely.Point(ni.pose.x,ni.pose.y))
        # min_node = ni
        
        return ni

    def getforwardsolutionvector(self,node_ref,node=None,tree_id=0):
        """0:forward tree, 1:backwardtree"""
        if tree_id ==0:
            solution_vec_ = []
            
            iter_node = node
            # print(iter_node.pose)
            if not iter_node == None:
                solution_vec_.append(iter_node.pose)

                iter_node = iter_node.parent
                # solution_vec_.reverse()
                # print(solution_vec_)
                while not iter_node is None :
                    # print("length of the parent path: ",len(iter_node.parent.path_from_parent))
                    solution_vec_.append(iter_node.pose)
                    iter_node = iter_node.parent

                    # connected_vec.pop(0)
                    # solution_vec_=solution_vec_+connected_vec
                    # iter_node = iter_node.parent
                # solution_vec_.append(iter_node.pose)
            # solution_vec_.reverse()
            next_node = node_ref
            next_node = next_node.parent

            next_fst_vec = []
            # next_fst_vec.reverse()
            # next_fst_vec.append(next_node.pose)
            # next_node = next_node.parent
            
            # solution_vec_ = solution_vec_ + next_fst_vec
            # print(node.pose)
            while not next_node is None:
                # print("length of the parent path: ",len(iter_node.parent.path_from_parent))
                next_fst_vec.append(next_node.pose)

                next_node = next_node.parent
            
            # next_fst_vec.append(next_node)
            next_fst_vec.reverse()
            solution_vec_=next_fst_vec+solution_vec_

        if tree_id ==1:
            solution_vec_ = []
            
            solution_vec_.append(self.goal_pos)
            # solution_vec_.reverse()
            next_node = node_ref
            # solution_vec_.append(shapely.Point(next_node.pose.x,next_node.pose.y,next_node.pose.z))
            
            # solution_vec_ = solution_vec_ + next_fst_vec
            # print(node.pose)
            while not next_node is None:
                solution_vec_.append(next_node.pose)
                next_node = next_node.parent
            solution_vec_.reverse()

        return solution_vec_
    
    def getbackwardsolutionvector(self,node_ref,node=None,tree_id=1):
        """0:forward tree, 1:backwardtree"""
        if tree_id ==1:
            solution_vec_ = []
            iter_node = node
            # print(iter_node.pose)
            # solution_vec_.append(iter_node.pose)
            # iter_node = iter_node.parent
            # print(solution_vec_)
            while not iter_node is None:
                # print("length of the parent path: ",len(iter_node.parent.path_from_parent))
                solution_vec_.append(iter_node.pose)
                
                iter_node = iter_node.parent
            
            # solution_vec_.append(iter_node.pose)
            solution_vec_.reverse()
            next_node = node_ref
            next_node = next_node.parent

            next_fst_vec = []
            # next_fst_vec.append(next_node.pose)
            # next_node = next_node.parent
            # print(node.pose)
            while not next_node is None:
                # print("length of the parent path: ",len(iter_node.parent.path_from_parent))
                next_fst_vec.append(next_node.pose)

                next_node = next_node.parent
            # next_fst_vec.append(next_node.pose)

            solution_vec_ = solution_vec_+next_fst_vec

        return solution_vec_
    
    def get_forward_final_trajectory(self):
        final_path = self.fsolutionvector
        return final_path
    
    def get_backward_final_trajectory(self):
        if self.bi_tree:
            final_path = self.bsolutionvector
        else:
            final_path = None
        return final_path

    def get_forward_child_state(self):
        random_x = random.uniform(self.bound[0],self.bound[2])
        random_y = random.uniform(self.bound[1],self.bound[3])

        # find the nearest point on the forward tree as the parent, choose 20 reference and delete all inactive from rtree
        parent_node = self.get_forward_nearest_point((random_x,random_y))
        if parent_node == None:
            # q = self.forward_tree_temp.nearest((random_x,random_y),1,objects=True)
            # parent_node = ctypes.cast(q[0].object, ctypes.py_object).value
            print("cannot find the nearest generated node, use root instead")
            parent_node = self.f_treeVector[0]
        forward_model = KinematicsNoAcc(parent_node.pose,self.wheel_base,self.map,[self.v_min,self.v_max],[self.steering_min,self.steering_max],self.min_time_steps,self.max_time_steps)
        

        forward_model_result = forward_model.forwardPropagate()
        if not forward_model_result:
            return False
        forward_result = [forward_model_result[0][-1],forward_model_result[1]]
        # print(forward_result)
        temp_cost = parent_node.cost+forward_result[1]

        if temp_cost+(shapely.distance(shapely.Point(forward_result[0].x,forward_result[0].y),shapely.Point(self.goal_pos.x,self.goal_pos.y))/self.v_max) < self.forward_best_cost_:
            # find the nearest circlenode 
            c_n = list(self.forward_circle_tree.nearest((forward_result[0].x,forward_result[0].y),1,objects=True))
            n = ctypes.cast(c_n[0].object, ctypes.py_object).value
            
            if shapely.distance(shapely.Point(forward_result[0].x,forward_result[0].y),shapely.Point(n.center.x,n.center.y))<self.circle_radius:
                # check whether the new state's cost better than the exist cost in the circle node
                n_obj = ctypes.cast(n.active_n, ctypes.py_object).value
                if temp_cost<n_obj.cost:
                    self.forward_snode_id+=1
                    # update the current circle node
                    n_obj.active = False
                    Pnodei = Node(self.forward_snode_id,forward_result[0],temp_cost,parent_node,[])
                    # print("forward_node_new path:", forward_model_result[0])

                    self.f_treeVector.append(Pnodei)
                    n.active_n =id(self.f_treeVector[self.forward_snode_id])
                    # with self.forward_lock:
                    # print("acquire the forward lock to change the forward tree")
                    self.forward_tree.delete(n_obj.id,(n_obj.pose.x,n_obj.pose.y))
                    self.forward_tree.insert(self.forward_snode_id,(Pnodei.pose.x,Pnodei.pose.y),id(self.f_treeVector[self.forward_snode_id]))
                    self.forward_last_node = self.f_treeVector[self.forward_snode_id]
                    # print("use new node in the search tree, delete the inactive node in the search tree")
                    # print("add a new node to the tree within the exist circle")
                else:
                    # self.get_forward_child_state()
                    return False
            else:
                self.forward_snode_id+=1
                self.forward_cnode_id+=1
                Pnodei = Node(self.forward_cnode_id,forward_result[0],temp_cost,parent_node,[])
                # print("forward_node_new path:", forward_model_result[0])
                self.f_treeVector.append(Pnodei)
                CPnodei = CircleNode(self.forward_cnode_id,Pnodei.pose,id(self.f_treeVector[self.forward_snode_id]))
                self.f_circleVector.append(CPnodei)
                self.forward_last_node = self.f_treeVector[self.forward_snode_id]

                # with self.forward_lock:
                # self.forward_lock.acquire()
                # print("acquire the forward lock to change the forward tree")
                self.forward_tree.insert(Pnodei.id,(Pnodei.pose.x,Pnodei.pose.y),id(self.f_treeVector[self.forward_snode_id]))
                self.forward_circle_tree.insert(CPnodei.id,(CPnodei.center.x,CPnodei.center.y),id(self.f_circleVector[self.forward_cnode_id]))
                # self.forward_lock.release()
                # print("add a new node to the tree and new circle node")
        else:
            # print(self.best_cost_)
            return False
        
        # check if the new_added node connected to the solution
        if self.bi_tree:
            # print("locked backward,use forward")
            # self.backward_lock.acquire()

            b_p = list(self.backward_tree.nearest((self.forward_last_node.pose.x,self.forward_last_node.pose.y),1,objects=True))
            b_n = ctypes.cast(b_p[0].object, ctypes.py_object).value
            dis_to_back = shapely.distance(shapely.Point(b_n.pose.x,b_n.pose.y),shapely.Point(self.forward_last_node.pose.x,self.forward_last_node.pose.y))
            if dis_to_back<=0.7:
                self.f_reach_end = True
                temp_best_cost = b_n.cost+self.forward_last_node.cost + dis_to_back/self.v_max
                if temp_best_cost < self.forward_best_cost_:
                    # print("path found from forward")


                    self.forward_best_cost_ = temp_best_cost
                    self.fsolutionvector = self.getforwardsolutionvector(node = b_n,node_ref = self.forward_last_node,tree_id=0)
                    # print(b_n.parent.path_to_parent)
            # self.backward_lock.release()
            if self.flag:
                self.get_backward_child_state()

            else:
                return True

        else:
            dis_to_goal = shapely.distance(shapely.Point(self.goal_pos.x,self.goal_pos.y),shapely.Point(self.forward_last_node.pose.x,self.forward_last_node.pose.y))
            if dis_to_goal <=0.7:
                self.f_reach_end = True
                # print("path found from forward")
                temp_best_cost = self.forward_last_node.cost
                if temp_best_cost < self.forward_best_cost_:
                    self.forward_best_cost_ = temp_best_cost
                    # self.forward_id+=1
                    # last_node = Node(self.forward_id,self.goal_pos,temp_best_cost)
                    self.fsolutionvector = self.getforwardsolutionvector(node_ref = self.forward_last_node,tree_id=1)
            
            if self.flag:
                return False
            else:
                return True




    def get_backward_child_state(self):
        random_x = random.uniform(self.bound[0],self.bound[2])
        random_y = random.uniform(self.bound[1],self.bound[3])

        # find the nearest point on the forward tree as the parent, choose 20 reference and delete all inactive from rtree
        parent_node = self.get_backward_nearest_point((random_x,random_y))
        if parent_node == None:
            # q = self.forward_tree_temp.nearest((random_x,random_y),1,objects=True)
            # parent_node = ctypes.cast(q[0].object, ctypes.py_object).value
            print("cannot find the nearest generated node, use goal instead")
            parent_node = self.b_treeVector[0]
        backward_model = KinematicsNoAcc(parent_node.pose,self.wheel_base,self.map,[self.v_min,self.v_max],[self.steering_min,self.steering_max],self.min_time_steps,self.max_time_steps)
        

        backward_model_result = backward_model.backwardPropagate()
        if not backward_model_result:
            return False
        backward_result = [backward_model_result[0][-1],backward_model_result[1]]
        temp_cost = parent_node.cost+backward_result[1]
        if temp_cost+(shapely.distance(shapely.Point(backward_result[0].x,backward_result[0].y),shapely.Point(self.start_pos.x,self.start_pos.y))/self.v_max) < self.backward_best_cost_:
            # find the nearest circlenode 
            c_n = list(self.backward_circle_tree.nearest((backward_result[0].x,backward_result[0].y),1,objects=True))
            n = ctypes.cast(c_n[0].object, ctypes.py_object).value
            
            if shapely.distance(shapely.Point(backward_result[0].x,backward_result[0].y),shapely.Point(n.center.x,n.center.y))<self.circle_radius:
                # check whether the new state's cost better than the exist cost in the circle node
                n_obj = ctypes.cast(n.active_n, ctypes.py_object).value
                if temp_cost<n_obj.cost:
                    self.backward_snode_id+=1
                    # update the current circle node
                    n_obj.active = False
                    Pnodei = Node(self.backward_snode_id,backward_result[0],temp_cost,parent_node,[])
                    # print("forward_node_new path:", backward_model_result[0])

                    self.b_treeVector.append(Pnodei)
                    n.active_n =id(self.b_treeVector[self.backward_snode_id])
                    # with self.forward_lock:
                    # print("acquire the forward lock to change the forward tree")
                    self.backward_tree.delete(n_obj.id,(n_obj.pose.x,n_obj.pose.y))
                    self.backward_tree.insert(self.backward_snode_id,(Pnodei.pose.x,Pnodei.pose.y),id(self.b_treeVector[self.backward_snode_id]))
                    self.backward_last_node = self.b_treeVector[self.backward_snode_id]
                    # print("use new node in the search tree, delete the inactive node in the search tree")
                    # print("add a new node to the back tree within the exist circle")
                else:
                    return False
            else:
                self.backward_snode_id+=1
                self.backward_cnode_id+=1
                Pnodei = Node(self.backward_cnode_id,backward_result[0],temp_cost,parent_node,[])
                # print("forward_node_new path:", backward_model_result[0])

                self.b_treeVector.append(Pnodei)
                CPnodei = CircleNode(self.backward_cnode_id,Pnodei.pose,id(self.b_treeVector[self.backward_snode_id]))
                self.b_circleVector.append(CPnodei)
                self.backward_last_node = self.b_treeVector[self.backward_snode_id]

                # with self.forward_lock:
                # self.backward_lock.acquire()
                # print("acquire the forward lock to change the forward tree")
                self.backward_tree.insert(Pnodei.id,(Pnodei.pose.x,Pnodei.pose.y),id(self.b_treeVector[self.backward_snode_id]))
                self.backward_circle_tree.insert(CPnodei.id,(CPnodei.center.x,CPnodei.center.y),id(self.b_circleVector[self.backward_cnode_id]))
                # self.backward_lock.release()
                # print("add a new node to the back tree and new circle node")
        else:
            # print(self.best_cost_)
            return False

        # self.forward_lock.acquire()
        f_p = list(self.forward_tree.nearest((self.backward_last_node.pose.x,self.backward_last_node.pose.y),1,objects=True))
        f_n = ctypes.cast(f_p[0].object, ctypes.py_object).value
        dis_to_forward = shapely.distance(shapely.Point(f_n.pose.x,f_n.pose.y),shapely.Point(self.backward_last_node.pose.x,self.backward_last_node.pose.y))
        if dis_to_forward<=0.7:
            self.b_reach_end = True
            temp_best_cost = f_n.cost+self.backward_last_node.cost + dis_to_forward/self.v_max
            if temp_best_cost < self.backward_best_cost_:
                # print("path found from backward")

                self.backward_best_cost_ = temp_best_cost
                self.bsolutionvector = self.getbackwardsolutionvector(node=f_n,node_ref = self.backward_last_node,tree_id=1)
        # self.get_forward_child_state()

        if self.flag:
            return False
        else:
            return True
        
        # self.forward_lock.release()

    def get_forward_tree(self):
        # current_time = time.time()
        complete_state = self.get_forward_child_state()
        while not complete_state:
            complete_state=self.get_forward_child_state()
            
            
            # current_time = time.time()
        # print("forward tree finished")

    def get_backward_tree(self):
        # current_time = time.time()
        while True:
            self.get_backward_child_state()
            if not self.flag:
                break
            # current_time = time.time()
        # print("backward tree finished")
        


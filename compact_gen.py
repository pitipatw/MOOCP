import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as etree
from svgpath2mpl import parse_path
from io import StringIO

# from linkage_utils import draw_mechanism

def compact_gen(N):
    x0 = dict()
    nc = 2
    #The first one will always be 1 (the motor)
    for i in range(2,N):
        c1,c2 = random.sample(list(np.arange(0,N)),nc)
        x0["C" + str(i) + "_0"] = c1
        x0["C" + str(i) + "_1"] = c2

    for i in range(2*N):
        x0["X0" + str(i)] = random.uniform(0,1)
    for i in range(N):
        x0["fixed_nodes"+str(i)] = 0
    
    x0["fixed_nodes0"] = 1 
    x0["fixed_nodes1"] = 1
    
    x0["target"] = 0
    return x0

def convert_x0_to_x(x0, N): 
    # N = self.N
    x = dict()

    #creating the upper triangluar x with all 0
    for i in range(N):
        for j in range(i):
            x["C" + str(j) + "_" + str(i)] = 0

    #Change only the one with the connection.
    nc = 2
    for i in range(2,N): 
        #randomly select nc nodes 
        c1 = x0["C" + str(i) + "_0"]
        c2 = x0["C" + str(i) + "_1"]
        #make sure c1 < c2 
        if c1 < c2:
            x["C"+str(c1)+"_"+str(c2)] = 1 
        else :
            x["C"+str(c2)+"_"+str(c1)] = 1


    for i in range(2*N):
        x["X0" + str(i)] = x0["X0" + str(i)]

    for i in range(N):
        x["fixed_nodes"+str(i)] = x0["fixed_nodes"+str(i)]
    # for i in range(N):
    #     x["fixed_nodes" + str(i)] = x0["fixed_nodes" + str(i)]
    x["target"] = x0["target"]
    
    return x


def convert_1D_to_mech(x,N):
    # N = self.N

    # Get target node value
    target = x["target"]

    # Build connectivity matrix from its flattened constitutive variables
    C = np.zeros((N,N))
    x["C0_1"] = 1

    for i in range(N):
        for j in range(i):
            C[i,j] = x["C" + str(j) + "_" + str(i)]
            C[j,i] = x["C" + str(j) + "_" + str(i)]

    # Reshape flattened position matrix to its proper Nx2 shape
    x0 = np.array([x["X0" + str(i)] for i in range(2*N)]).reshape([N,2])

    # Extract a list of Nodes that are fixed from boolean fixed_nodes vector
    fixed_nodes = np.where(np.array([x["fixed_nodes" + str(i)] for i in range(N)]))[0].astype(int)

    #We fix the motor and original ground node as 0 and 1 respectively in this implementation
    motor=np.array([0,1])

    target = 0 
    return target, C, x0, fixed_nodes, motor


def draw_mechanism(A,x0,fixed_nodes,motor, highlight=100, solve=True, thetas = np.linspace(0,np.pi*2,200), def_alpha = 1.0, h_alfa =1.0, h_c = "#f15a24"):
    
    # valid, _, _, _ = solve_mechanism(A, x0, fixed_nodes, motor, device = "cpu", timesteps = 2000)
    valid = True
    if not valid:
        print("Mechanism is invalid!")
        return

    fig = plt.figure(figsize=(12,12))

    def fetch_path():
        root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    


    A,x0,fixed_nodes,motor = np.array(A),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    
    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=700,zorder=10,marker=p)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=100,zorder=10,facecolors=h_c,alpha=0.7)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=100,zorder=10,facecolors='#ffffff',alpha=0.7)
        
        for j in range(i+1,N):
            if A[i,j]:
                if (motor[0] == i and motor[1] == j) or(motor[0] == j and motor[1] == i):
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#ffc800",linewidth=4.5)
                else:
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#1a1a1a",linewidth=4.5,alpha=0.6)
                
    if solve:
        path = find_path(A,motor,fixed_nodes)[0]
        G = get_G(x0)
        x,c,k =  solve_rev_vectorized(path.astype(int), x0, G, motor, fixed_nodes,thetas)
        x = np.swapaxes(x,0,1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
            plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    plt.axis('equal')
    plt.axis('off')
print(compact_gen(5))

x0 = compact_gen(5)
x = convert_x0_to_x(x0,5)
target, C, x0, fixed_nodes, motor = convert_1D_to_mech(x, 5)

draw_mechanism(C,x0,fixed_nodes,motor)

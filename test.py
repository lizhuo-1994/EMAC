import numpy as np

lower_bound = np.array([-1,-1])
upper_bound = np.array([1,1])

k = 2

def con_2_dis(con_states):
    
    unit = (upper_bound - lower_bound)/k
    abs_states = np.zeros(con_states.shape[0],dtype=np.int8)
        
    #print(lower_bound)
    #print(upper_bound)
    indixes = np.where(unit == 0)[0]
    unit[indixes] = 1
    #print('unit:\t', unit)
        
    tmp = ((con_states-lower_bound)/unit).astype(int)

            
    dims = tmp.shape[1]
    for i in range(dims):
        abs_states = abs_states + tmp[:,i]*pow(k, i)
#         abs_states = np.expand_dims(abs_states,axis=-1)
            
    abs_states = [str(item) for item in abs_states]
    return abs_states

con_states = np.array([[0.99,0.1]])
print(con_2_dis(con_states))
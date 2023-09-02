from elephant.spike_train_dissimilarity import victor_purpura_distance
import quantities as pq
from neo import SpikeTrain
import numpy as np 
from tqdm import tqdm

def spkd(tli, tlj, cost=1):

    """
    Converted to python from http://www-users.med.cornell.edu/~jdvicto/spkdm.html

    %
    % d=spkd(tli,tlj,cost) calculates the "spike time" distance
    % (Victor & Purpura 1996) for a single cost
    %
    % tli: vector of spike times for first spike train
    % tlj: vector of spike times for second spike train
    % cost: cost per unit time to move a spike
    %
    %  Copyright (c) 1999 by Daniel Reich and Jonathan Victor.
    %  Translated to Matlab by Daniel Reich from FORTRAN code by Jonathan Victor.
    %
    """
    
    # Length of the spike train
    nspi = len(tli)
    nspj = len(tlj)

    if cost==0:
        # Just a difference in spike numbers
        d = abs(nspi-nspj)
        return d
    # elif cost==Inf:
    #     d = nspi+nspj
    #     return d

    scr = np.zeros((nspi+1, nspj+1))
    # %
    # %     INITIALIZE MARGINS WITH COST OF ADDING A SPIKE
    # %
    scr[:,0]  = np.arange(nspi+1)
    scr[0,:]  = np.arange(nspj+1)

    for i in range(1, nspi+1):
        for j in range(1, nspj+1):
            scr[i,j] = min( [scr[i-1,j]+1,  scr[i,j-1]+1,  scr[i-1,j-1]+cost*abs(tli[i-1]-tlj[j-1])] )
        
    d=scr[nspi, nspj]

    return d

def compare():

    """

    Compare two implementations of Victor-Purpura distance.

    """

    print("Computing distances ...")

    sampling_rate = 30

    # DEMO - VP distance of spkd implementation
    
    # add one step 1/sampling_rate --> diff is one time step
    tli, tlj = [1, 4], [1, 4+1/sampling_rate]
    print(tli, tlj)
    
    # DEMO - VP distance of elephant library

    # set step size
    q = 1 / ( 1/sampling_rate * pq.s)
    
    st_a = SpikeTrain(tli, units='s', t_stop= 10.0)
    st_b = SpikeTrain(tlj, units='s', t_stop= 10.0)
    vp_f = victor_purpura_distance([st_a, st_b], q)
    print(spkd(tli,tlj, cost=sampling_rate), vp_f[0,1])

    # add value smaller then 1/sampling_rate --> diff is step cost/2
    tli, tlj = [1, 4], [1, 4+1/(sampling_rate*2)]
    print(tli, tlj)

    q = 1 / ( 1/sampling_rate * pq.s)
    
    st_a = SpikeTrain(tli, units='s', t_stop= 10.0)
    st_b = SpikeTrain(tlj, units='s', t_stop= 10.0)
    vp_f = victor_purpura_distance([st_a, st_b], q)
    
    print(spkd(tli,tlj, cost=sampling_rate), vp_f[0,1])
    
    # Check with random generated numbers
    for i in tqdm(range(10)):
        sampling_rate = 30
        
        tli, tlj = np.sort(np.random.random(5)*10), np.sort(np.random.random(5)*10)
        
        dist = spkd(tli, tlj, cost=sampling_rate)

        q = 1 / ( 1/sampling_rate * pq.s)
  
        st_a = SpikeTrain(tli, units='s', t_stop= 1000.0)
        st_b = SpikeTrain(tlj, units='s', t_stop= 1000.0)
        vp_f = victor_purpura_distance([st_a, st_b], q)
        
        np.testing.assert_almost_equal(dist, vp_f[0, 1], decimal=5, err_msg='two implementations are not equal!')

if __name__ == "__main__":
    compare()




  
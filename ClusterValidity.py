import os
import copy
import math
import random
import sklearn
import scipy
import itertools
import sklearn.cluster as clstr
import sklearn.mixture as mxtr
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from functools import reduce
from numpy.linalg import norm
#from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
#from sklearn.mixture import GMM, DPGMM 

class ClusterVal(object):
    # self.X ; self.c ; self.labels ; self.scatter_matrix ; self.TSS ; self.barycenter ; self.barycenters ; self.clusters ; self.within_group_scatter ; 
    # self.within_cluster_dispersion ; 
    def __init__(self,X,c=None,remember=True):
        # X = n x d sample matrix; c = n x 1 clustering matrix whose values range from 1 to k, k the predefined number of clusters
        if remember==True:
            print("Remember to use 'vars(this_object)' to check all variables belonging to this object!!")
        if type(X) != type(np.array([1])):
            self.X = np.array(X)
        else:
            self.X = copy.copy(X)
        
        if c is not None:
            if len(c)!=self.X.shape[0]:
                print("length of c should match height of X!")
            else:
                if type(c)!= type(np.array([1])):
                    self.c = np.array(c)
                else:
                    self.c = copy.copy(c)

    def ListVariables(self):
        return vars(self).keys()

    def SetMatrix(self,X):
    # Use SetMatrix for properly setting of self.X parameter! (copying garantees new allocation in memory!)
    # Also, this method garantees that self.X is an numpy.array !
        if type(X) != type(np.array([1])):
            self.X = np.array(X)
        else:
            self.X = copy.copy(X)

    def SetLabels(self,c):
    # Use SetLabels for properly setting of self.c parameter! (copying garantees new allocation in memory!)
    # Also, this method garantees that self.c is an numpy.array, and sets the labels's set!
        if type(c) != type(np.array([1])):
            self.c = np.array(c)
        else:
            self.c = copy.copy(c)
        self.labels = set(self.c)

    def ScatterMatrix(self, set_it=False):
    # ScatterMatrix calculates the scatter matrix of self.X (variance-covariance matrix).
        A = self.X - np.mean(self.X,axis=0)
        A = (A.T).dot(A)
        if set_it==False:
            return A
        else:
            self.scatter_matrix = A

    def TSS(self,set_it=False):
    # TSS calculates the Total Sum of Squares of the scatter matrix of self.X (variance-covariance matrix), which is its trace.
        A = self.X - np.mean(self.X,axis=0)
        tss = np.trace(A.T.dot(A))
        if set_it==False:
            return tss
        else:
            self.TSS = tss

    def Barycenter(self,set_it=True):
    # Barycenter sets or return the barycenter -as a numpy.array- of all the samples in self.X .
        if set_it==True:
            self.barycenter = np.mean(self.X,axis=0)
        else:
            return np.mean(self.X,axis=0)
    
    def ClustersBarycenters(self,set_it=True):
    # ClustersBarycenters returns or sets a list of barycenters, one for each label among those in self.c
        if self.c is None:
            print("c in empty, you shall fill it!")
        else:
            labels = list(set(self.c))
            if set_it==True:
                self.barycenters = []
                for l in labels:
                    self.barycenters.append((l,np.mean(self.X[self.c==l],axis=0)))
            else:
                barycenters = []
                for l in labels:
                    barycenters.append((l,np.mean(self.X[self.c==l],axis=0)))
                return barycenters

    def SetClusters(self):
        labels = list(set(self.c))
        self.clusters = []
        for l in labels:
            self.clusters.append(self.X[self.c==l])
        print("Done! self.clusters is now setted!")

    def WG(self,set_it=True):
    # WG calculates the Within-group scatter. Returns or sets it as a list of tuples where first term is the cluster label and
    # and second term is its variance-covariance matrix
    # Output = [(label_0,WG_0),(label_1,WG_1), ... ] where WG_k = X_k.T,dot(X_k), X_k a centered matrix around k-th barycenter of the k-th cluster
        if self.c is None:
            print("c is empty, you shall fill it!")
        elif len(self.c)!=self.X.shape[0]:
            print("c and X should match their dimension!")
        else:
            labels = list(set(self.c))
            if set_it==True:
                self.within_group_scatter = []
                for l in labels:
                    A = self.X[self.c==l]
                    A = A - np.mean(A,axis=0)
                    self.within_group_scatter.append((l,(A.T).dot(A)))
                s = self.within_group_scatter[0][1].shape
                self.wg = np.zeros(s)
                for x in self.within_group_scatter:
                    self.wg += x[1]
                
            else:
                within_group_scatter = []
                for l in labels:
                    A = self.X[self.c==l]
                    A = A - np.mean(A,axis=0)
                    within_group_scatter.append((l,(A.T).dot(A)))
                s = within_group_scatter[0][1].shape
                #s = self.X.shape[1]
                wg = np.zeros(s)
                for x in within_group_scatter:
                    wg += x[1]
                return within_group_scatter,wg

    def WGSS(self,set_it=True):
    # WGSS calculates the Within-group dispersion and the pooled within-cluster sum of squares(WGSS stands for Within-group sum of squares). 
        if self.c is None:
            print("c is empty, you shall fill it!")
        elif len(self.c)!=self.X.shape[0]:
            print("c and X should match their dimension!")
        else:
            labels = list(set(self.c))
            if set_it==True:
                self.within_cluster_dispersion = []
                self.wgss = np.array([0],dtype=np.float64)
                #self.wgss = 0
                for l in labels:
                    A = self.X[self.c==l]
                    A = A - np.mean(A,axis=0)
                    A = (A.T).dot(A)
                    self.within_cluster_dispersion.append((l,np.trace(A)))
                    self.wgss += np.trace(A).astype(np.float64)
                    #self.wgss += np.trace(A)
                    self.wgss = float(self.wgss)
            else:
                within_cluster_dispersion = []
                wgss = 0 #np.array([0],dtype=np.float64)
                for l in labels:
                    A = self.X[self.c==l]
                    A = A - np.mean(A,axis=0)
                    A = (A.T).dot(A)
                    within_cluster_dispersion.append((l,np.trace(A)))
                    wgss += np.trace(A) #.astype(np.float64)
                return within_cluster_dispersion,float(wgss)
  
    def ClustersSizes(self,set_it=True):
        if self.c is None:
            print("c in empty, you shall fill it!")
        else:
            labels = list(set(self.c))
            sizes = []
            for l in labels:
                sizes.append((l,self.X[self.c==l].shape[0]))
            if set_it==True:
                self.sizes=sizes
            else: 
                return sizes

    def ClustersDiameters(self,diameter=1,mu=None,d='Euclid',set_it=True):
    # ClusterDiameters calculates the diameters of the given clusters. Of course its calculations depends on he many different notions
    # of 'diameter' and the distance function used (here the default is the euclidean). The parameter diameter can take three values:
    #
    # diameter = 1 => diam(C_i) = max d(x,y) , which calculates the maximum distance among all pairs in each cluster;
    # diameter = 2 => diam(C_i) = 1/(|C_i||C_i-1|)*sum_{C_i}{d(x,y)}, which calculates the mean distance between all pairs;
    # diameter = 3 => diam(C_i) = 2/|C_i|*sum_{C_i}{d(x,u)}, which calculates twice (* 2 !!!) the mean distance of all the points from the mean u.
    #
    # if mu=None => u = 1/|C_i|*sum_{C_i}{x} else, u should be given so that u == mu[l] where mu is passed as a dict and l are labels values
    # else mu == {l_0:u_0 , l_1:u_1 , ... , l_k:u_k} 
        if self.c is None:
            print("c in empty, you shall fill it!")
        else:
            labels = list(set(self.c))
            
            if diameter==1:
                diameters = []
                for l in labels:
                    A = self.X[self.c==l]
                    distances = []
                    
                    if d=='Euclid':
                        s = np.linalg.norm
                        for i,x in enumerate(A):
                            for j,y in enumerate(A):
                                if i < j:
                                    distances.append(s(x-y))
                    else: # d = lambda x,y : some_distance_function(x-y)
                        for i,x in enumerate(A):
                            for j,y in enumerate(A):
                                if i < j:
                                    distances.append(d(x,y))
                                    
                    diameters.append((l,max(distances)))

            elif diameter==2:
                diameters = []
                for l in labels:
                    A = self.X[self.c==l]
                    n_l = A.shape[0]
                    n_l = n_l*(n_l-1)
                    diameter = 0
                    
                    if d=='Euclid':
                        s = np.linalg.norm
                        for i,x in enumerate(A):
                            for j,y in enumerate(A):
                                if i < j:
                                    diameter += s(x-y)
                    else:
                        for i,x in enumerate(A):
                            for j,y in enumerate(A):
                                if i < j:
                                    diameter += d(x,y)
                                    
                    diameters.append((l,diameter/n_l))                
            
            elif diameter==3:
                diameters = []
                for l in labels:
                    A = self.X[self.c==l]
                    n_l = A.shape[0]

                    if mu is None:
                        u = np.mean(A,axis=0)
                    else:
                        u = mu[l]

                    diameter = 0
                    if d=='Euclid':
                        s = np.linalg.norm
                        for x in A:
                            diameter += s(x-u)
                    else:
                        for x in A:
                            diameter += d(x,u)
                            
                    diameters.append((l,2*diameter/n_l))
                    # P.S.: here we multiply by 2 the mean distance from the centroid of the cluster, as this represents a diameter!
            
            if set_it==True:
                self.diameters = diameters
            else:
                return diameters

    def BetweenGroupScatter(self,set_it=True):
    # Calculates the Between-group scatter of a cluster set
        if self.c is None:
            print("c in empty, you shall fill it!")
        else:
            things_to_stack = []
            barycenters = self.ClustersBarycenters(set_it=False)
            m = self.Barycenter(set_it=False)
            for l,u in barycenters:
                n_l = self.X[self.c==l].shape[0]
                things_to_stack.append(np.outer(np.ones((1,n_l)),u - m))
            B = np.vstack(things_to_stack)
            if set_it==True:
                self.bg = B.T.dot(B)
            else:
                return B.T.dot(B)
 
    def BGSS(self,set_it=True,np_direct=True):
    # Calculates the between-group dispersion of a cluster set, named BGSS (stands for between-group sum of squares), 
    # which is the trace of the between-group scatter. Here it is calculated in a different - robust - manner, 
    # although the direct numpy calculation of the trace can be performed through the variable np_direct = True
        if self.c is None:
            print("c is empty you shall fill it!")
        else:
            if np_direct==False:
                m = self.Barycenter(set_it=False)
                barycenters = self.ClustersBarycenters(set_it=False)
                bgss = 0
                for l,u in barycenters:
                    n_l = self.X[self.c==l].shape[0]
                    d_l = np.linalg.norm(u-m)
                    bgss += n_l*(d_l**2) #d_l*d_l
                if set_it==True:
                    self.bgss = bgss
                else:
                    return bgss
                    
            else:
                bg = self.BetweenGroupScatter(set_it=False)
                if set_it==True:
                    self.bgss = np.trace(bg)
                else:
                    return np.trace(bg)
 
    def PairNumbers(self,set_it=True):
        labels = list(set(self.c))
        N_w = 0
        for l in labels:
            n_l = self.X[self.c==l].shape[0]
            N_w += n_l*(n_l-1)/2
        
        N_b = 0
        t = list(zip(labels,range(len(labels))))
        for x in t:
            for y in t:
                if x[1]< y[1]:
                
                    N_b += x[0]*y[0]
        
        N_t = self.X.shape[0]
        N_t = N_t*(N_t - 1)/2
        
        if set_it==True:
            sizes = self.ClustersSizes(set_it=False)

    def VectorConcordance(self,A,B):
        a = np.array([1])
        if type(A)==type(a):
            X = copy.copy(A)
        else:
            X = np.array(A)
        if type(B)==type(a):
            Y = copy.copy(B)
        else:
            Y = np.array(B)
        if X.shape != Y.shape:
            print("A and B shapes should match!")
        else:
            X = X.flatten()
            Y = Y.flatten()
            
            n = X.shape[0]
            
            concordances_dict = dict()
            concordances_list = []
            for pair in itertools.combinations(range(n),2):
                x = X[pair[0]]<X[pair[1]]
                y = Y[pair[0]]<Y[pair[1]]
                concordances_dict[pair] = (x,y)
                concordances_list.append( (x and y)or(not x and not y) )
            
            S_plus = len([x for x in concordances_list if x==True])
            S_minus = len([x for x in concordances_list if x==False])
            
            concordances_dict[0] = (S_plus,S_minus)
            
            return concordances_dict

    def GammaIndex(self,A,B):
        d = self.VectorConcordance(A,B)
        gamma_index = (d[0][0] - d[0][1])/(d[0][0] + d[0][1])
        return gamma_index
            
    def BallHall(self,set_it=True):
    # The mean dispersion of a cluster is the mean of the squared distances of the points 
    # of the cluster with respect to their barycenter. The Ball-Hall index is the mean, 
    # through all the clusters, of their mean dispersion.
        if self.c is None:
            print("c in empty, you shall fill it!")
        else: 
            barycenters = self.ClustersBarycenters(set_it=False)
            ball_hall=0
            for l,u in barycenters:
                D = self.X[self.c==l] - u
                n_l = D.shape[0]
                ball_hall += np.trace(D.dot(D.T))/n_l
            ball_hall /= len(set(self.c))
            if set_it==True:
                self.ball_hall = ball_hall
            else:
                return ball_hall 
    
    def BanfeldRaftery(self,set_it=True):
    # This index is the weighted sum of the logarithms of the traces of the 
    # variance-covariance matrix of each cluster. 
        if self.c is None:
                print("c in empty, you shall fill it!")
        else:
            within_group_scatter,wg = self.WG(set_it=False)
            banfeld_raftery=0
            for l,A in within_group_scatter:
                n_l = self.X[self.c==l].shape[0]
                banfeld_raftery += n_l*np.log(np.trace(A)/n_l)
            if set_it==True:
                self.banfeld_raftery = banfeld_raftery
            else:
                return banfeld_raftery 
    
    def CIndex(self,d='Euclid',set_it=True):
    # CIndex implements the ratio (S_w - S_min)/(S_max - S_min) in which S_w is the sum of the N_w distances between all the pairs of points inside each cluster;
    # S_min is the sum of the N_w smallest distances between all the pairs of points in the entire data set. There are N_t such pairs: one takes the sum of the
    # N_w smallest values ; S_max is the sum of the N_w largest distances between all the pairs of points in the entire data set. There are N_t such pairs: 
    # one takes the sum of the N_w largest values.
    # As the indices S need the calculation of a (suitable) distance, this function it is left as optional, whose default is the euclidean norm.
        if self.c is None:
                print("c in empty, you shall fill it!")
        else:
            labels = list(set(self.c))
            N_w = 0
            S_w = 0
            for l in labels:
                A = self.X[self.c==l]
                n_l = A.shape[0]
                N_w += n_l*(n_l-1)/2

                if d=='Euclid':
                    s = np.linalg.norm
                    combinations = itertools.combinations(A,2)
                    for x in combinations:
                        S_w += s(x[0] - x[1])
                    #for i,x in enumerate(A):
                     #   for j,y in enumerate(A):
                      #      if i < j:
                       #         S_w += s(x-y)
                else:   # d = lambda x,y: some_distance_function(x-y)
                    combinations = itertools.combinations(A,2)
                    for x in combinations:
                        S_w += s(x[0] - x[1])
                    #for i,x in enumerate(A):
                     #   for j,y in enumerate(A):
                      #      if i < j:
                       #         S_w += d(x,y)

            N_w = int(N_w)
            
            if d=='Euclid':
                s = np.linalg.norm
                distances = []
                combinations = itertools.combinations(self.X,2)
                for x in combinations:
                    distances.append(s(x[0] - x[1]))
                #for i,x in enumerate(self.X):
                 #   for j,y in enumerate(self.X):
                  #      if i < j:
                   #         distances.append(s(x-y))
            else:   # d = lambda x,y: some_distance_function(x-y)
                distances = []
                combinations = itertools.combinations(self.X,2)
                for x in combinations:
                    distances.append(d(x[0],x[1]))
                #for i,x in enumerate(self.X):
                 #   for j,y in enumerate(self.X):
                  #      if i < j:
                   #         distances.append(d(x,y))
                        
            distances.sort()
            S_min = sum(distances[:N_w])
            distances.sort(reverse=True)
            S_max = sum(distances[:N_w])
            
            c_index = (S_w - S_min)/(S_max - S_min)

            if set_it==True:
                self.c_index = c_index
            else:
                return c_index 
                
    def CalinskiHarabasz(self,set_it=True):
    # CalinskiHarabasz returns the value of ( (N −K)/(K −1) ) . ( (BGSS)/(WGSS) )
        if self.c is None:
            print("c in empty, you shall fill it!")
        else:
            K = len(set(self.c))
            N = self.X.shape[0]
            BGSS = self.BGSS(set_it=False)
            WGSS = self.WGSS(set_it=False)[1]
            calinski_harabasz = ((N-K)/(K-1))*((BGSS)/(WGSS))

            if set_it==True:
                self.calinski_harabasz = calinski_harabasz
            else:
                return calinski_harabasz

    def DunnIndex(self,diameter=1,mu=None,d='Euclid',dist='Euclid',set_it=True):
        if self.c is None:
            print("c is empty, you shall fill it!")
        else:
            labels = list(set(self.c))
            if mu is None:
                mu = dict()
                for l in labels:
                    mu[l] = np.mean(self.X[self.c==l],axis=0)
                    
            diameter = self.ClustersDiameters(diameter=diameter,mu=mu,d=d,set_it=False)
            
            if dist=='Euclid':
                delta = [np.linalg.norm(mu[x[0]] - mu[x[1]]) for x in itertools.combinations(labels,2)]
            else:
                delta = [dist(mu[x[0]],mu[x[1]]) for x in itertools.combinations(labels,2)]
                
            diameter = [x[1] for x in diameter]
            
            delta = min(delta)
            diameter = max(diameter)
            
            dunn_index = delta/diameter
            
            if set_it==True:
                self.dunn_index = dunn_index
            else:
                return dunn_index

    def BakerHubart(self,dist='Euclid',set_it=True):
        if self.c is None:
            print("c is empty, u shall fill it!")
        else:
            n = self.X.shape[0]
            if dist=='Euclid':
                distances = [np.linalg.norm(self.X[x[0]] - self.X[x[1]]) for x in itertools.combinations(range(n),2)]
            else:
                distances = [dist(self.X[x[0]],self.X[x[1]]) for x in itertools.combinations(range(n),2)]
            
            A = np.array(distances) #del distances
            B = [1 if self.c[x[0]]==self.c[x[1]] else 0 for x in itertools.combinations(range(n),2) ]
            
            baker_hubart = self.GammaIndex(A,B)
            
            if set_it==True:
                self.baker_hubart = baker_hubart
            else:
                return baker_hubart

    def DetRatio(self,set_it=True):
        if self.c is None:
            print("c is empty, u shall fill it!")
        else:
            wgs,wg = self.WG(set_it=False)
            t = self.ScatterMatrix(set_it=False)
            
            det_ratio = np.det(t)/np.det(wg)
            
            if set_it==True:
                self.det_ratio = det_ratio
            else:
                return det_ratio

    def LogDetRatio(self,set_it=True):
        if self.c is None:
            print("c is empty, u shall fill it!")
        else:
            wgs,wg = self.WG(set_it=False)
            t = self.ScatterMatrix(set_it=False)
            N = self.X.shape[0]
            
            log_det_ratio = N*np.log(np.linalg.det(t)/np.linalg.det(wg))
            
            if set_it==True:
                self.log_det_ratio = log_det_ratio
            else:
                return log_det_ratio

    def LogSSRatio(self,set_it=True,np_direct=True):
        if self.c is None:
            print("c is empty, u shall fill it!")
        else:
            wcd,wgss = self.WGSS(set_it=False)
            bgss = self.BGSS(set_it=False,np_direct=np_direct)
            
            log_ss_ratio = np.log(bgss/wgss)
            
            if set_it==True:
                self.log_ss_ratio = log_ss_ratio
            else:
                return log_ss_ratio

    def DaviesBouldin(self,dist='Euclid',set_it=True):
        if self.c is None:
            print("c is empty, u shall fill it!")
        else:
            if dist=='Euclid':
                d = np.linalg.norm

            barycenters = dict()
            labels = list(set(self.c))
            for l in labels:
                barycenters[l] = np.mean(self.X[self.c==l],axis=0)

            deltas = dict()
            for l in iter(barycenters):
                C = self.X[self.c==l] - barycenters[l]
                C = d(C,axis=1)
                deltas[l] = C.mean(axis=0)
            
            davies_bouldin = dict()
            for l in iter(barycenters):
                if dist=='Euclid':
                    davies_bouldin[l] = max([ (deltas[l] + deltas[k])/d(barycenters[l] - barycenters[k]) for k in iter(barycenters) if k!=l])
                else:
                    davies_bouldin[l] = max([ (deltas[l] + deltas[k])/dist(barycenters[l],barycenters[k]) for k in iter(barycenters) if k!=l])
            
            davies_bouldin = sum(davies_bouldin.values())/len(davies_bouldin)
            
            if set_it==True:
                self.davies_bouldin = davies_bouldin
            else:
                return davies_bouldin 

    def Silhouette(self,dist='Euclid',set_it=True):
        if self.c is None:
            print("c is empty, u shall fill it!")
        else:
            if dist=='Euclid':
                d = np.linalg.norm

            
            labels = list(set(self.c))

            s = dict()
            n = self.X.shape[0]
            for i in range(n):
                color = self.c[i]
                if dist=='Euclid':
                    d = np.linalg.norm
                    a = np.array([ d(self.X[i] - self.X[j]) for j in np.arange(n)[self.c==color] if j != i ])
                    a = np.mean(a)
                    b = dict()
                    for l in labels:
                        if l!=color:
                            b[l] = np.array([ d(self.X[i] - x) for x in self.X[self.c==l] ])
                    b = [np.mean(b[l]) for l in iter(b)]
                    b = min(b)
                else:
                    a = np.array([ dist(self.X[i],self.X[j]) for j in np.arange(n)[self.c==color] if j != i ])
                    a = np.mean(a)
                    b = dict()
                    for l in labels:
                        if l!=color:
                            b[l] = np.array([ dist(self.X[i],x) for x in self.X[self.c==l] ])
                    b = [np.mean(b[l]) for l in iter(b)]
                    b = min(b)
                    
                s[i] = (b-a)/max(a,b)
                
            clusters_silhouettes = dict()
            for l in labels:
                s_l = np.array([s[i] for i in np.arange(n)[self.c==l]])
                clusters_silhouettes[l] = np.mean(s_l)
            
            silhouette = np.array([clusters_silhouettes[i] for i in iter(clusters_silhouettes)])
            silhouette = np.mean(silhouette)

            if set_it==True:
                self.silhouette,self.clusters_silhouettes = silhouette,clusters_silhouettes
            else:
                return silhouette,clusters_silhouettes

    def Test(self,**kwargs):        

        cluster_method = kwargs['cluster_method']
        cluster_params = kwargs['cluster_params']
        
        self.SetLabels(cluster_method(*cluster_params))
        
        return [self.CIndex(set_it=False),self.CalinskiHarabasz(set_it=False),self.DunnIndex(set_it=False),self.BakerHubart(set_it=False),
                    self.DaviesBouldin(set_it=False),self.Silhouette(set_it=False)]
    
    def BigKMeansTest(self,X=None,k_min=4,k_max=20,**kwargs):
        
        indices = []
        for k in range(k_min,k_max+1):
            if X!=None:
                self.SetMatrix(X)
            else:
                clus = Clustering(self.X).KMeans
                params = [k,None,False]
                indices.append(self.Test(cluster_method=clus,cluster_params=params))
        inds = ['c index','Calinski','Dunn','Baker','Davies', 'Silhouette']
        return pd.DataFrame(indices,columns=inds)
        
    def KMeansTest(self,X=None,k_min=4,k_max=20):
        
        indices = []
        
        if X!=None:
            self.SetMatrix(X)
        else:
            indices = []
            clstr = Clustering(self.X)
            for k in range(k_min,k_max+1):
                print(k)
                self.c = clstr.KMeans(k,set_it=False)
                print(self.c)
                tests = [self.CIndex(set_it=False),self.CalinskiHarabasz(set_it=False),self.DunnIndex(set_it=False),self.BakerHubart(set_it=False),self.DaviesBouldin(set_it=False),self.Silhouette(set_it=False)]
                print(tests[0])
                
                indices.append(tests)
        inds = ['c index','Calinski','Dunn','Baker','Davies', 'Silhouette']
        return pd.DataFrame(indices,columns=inds)

class Clustering(object):

    def __init__(self,X):
        
        if type(X) != type(np.array([1])):
            self.X = np.array(X)
        else:
            self.X = copy.copy(X)

    def SetMatrix(self,X):
        if type(X) != type(np.array([1])):
            self.X = np.array(X)
        else:
            self.X = copy.copy(X)
    
    def PCA(self,dim,X=None,pca=2,set_it=True):
    # basically, PCA picks the first k = dim relevant dimensions through SVD, 
    # and we reduce the high-dimensional vectors into these k dimensions
    # D = Data for clustering (a copy of input X)
    # pca==1: it doesn't centralize D
    # pca==2: it does normalize D, the canonical PCA algorithm
        if (X is None)&(self.X is None):
            print("Set the matrix of features")
        elif (X is None)&(self.X is not None):
            D = copy.copy(self.X)
        elif X is not None:
            if type(X)==type(np.array([1])):
                D = copy.copy(X)
            elif type(X)==type(pd.DataFrame([1])):
                D = np.array(X)
            elif (type(X)!=type(np.array([1])))&(type(X)!=type(pd.DataFrame([1]))):
                try:
                    D = np.array(X)
                except:
                    print('Ops! Matrix X is not an accepted type')            

        if pca==1:
            U,S,V = np.linalg.svd(D.T)
            D = V[:dim,:].T
        elif pca==2:
            #u = D.mean(axis=0)       
            #S = (D-u).T.dot(D-u)
            D -= D.mean(axis=0)
            S = D.T.dot(D)
            U,W,V = np.linalg.svd(S)
            D = (V[:dim,:].dot(D.T)).T

        if set_it==True:
            self.X_pca = D
        else:
            return D
    
    def KMeans(self,k,X=None,set_it=True):
    
        kmeans = clstr.KMeans(n_clusters=k)
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            colors = kmeans.fit_predict(self.X)
        elif X is not None:
            if type(X)==type(np.array([1])):
                colors = kmeans.fit_predict(X)
            elif type(X)==type(pd.DataFrame([1])):
                colors = kmeans.fit_predict(np.array(X))
            elif (type(X)!=type(np.array([1])))&(type(X)!=type(pd.DataFrame([1]))):
                try:
                    colors = kmeans.fit_predict(X)
                except:
                    print('Ops! Matrix X is not an accepted type')
        
        if set_it==True:
            self.kmeans = colors
        else:
            return colors
 
    def GaussianMixture(self,k,X=None,init=10,set_it=True):
        
        gmm = mxtr.GMM(n_components=k,n_init=init)
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            colors = gmm.fit_predict(self.X)
        elif X is not None:
            colors = gmm.fit_predict(X)
        
        if set_it==True:
            self.gmm = colors
        else:
            return colors
        
    def DPGaussianMixture(self,k,X=None,alpha=1.0,tol=1e-3,n_iter=10,set_it=True):

        dpgmm = mxtr.DPGMM(n_components=k,alpha=alpha,tol=tol,n_iter=n_iter)
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            colors = dpgmm.fit_predict(self.X)
        elif X is not None:
            colors = dpgmm.fit_predict(X)
        
        if set_it==True:
            self.dpgmm = colors
        else:
            return colors

    def DBSCAN(self,X=None,eps=0.5,min_samples=10,set_it=True):
        dbscan = clstr.DBSCAN(eps=0.5,min_samples=10)
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            colors = dbscan.fit_predict(self.X)
        elif X is not None:
            colors = dbscan.fit_predict(X)
        
        if set_it==True:
            self.dbscan = colors
        else:
            return colors
    
    def AgglomerativeClustering(self,k,X=None,set_it=True):
        agglomerative = clstr.AgglomerativeClustering(n_clusters=k)
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            colors = agglomerative.fit_predict(self.X)
        elif X is not None:
            colors = agglomerative.fit_predict(X)
        
        if set_it==True:
            self.agglomerative_clustering = colors
        else:
            return colors

    def SpectralClustering(self,k,X=None,set_it=True,n_init=10, gamma=1.0, assign_labels='kmeans'):
        spectral_clustering = clstr.SpectralClustering(n_clusters=k,n_init=n_init,gamma=gamma,assign_labels=assign_labels)
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            colors = spectral_clustering.fit_predict(self.X)
        elif X is not None:
            colors = spectral_clustering.fit_predict(X)
        
        if set_it==True:
            self.spectral_clustering = colors
        else:
            return colors
        
    def Orclus(self,k,l,X=None,alpha=0.7,delta=10,dist='Euclid'):
        if (X is None)&(self.X is None):
            print("Set sample matrix!!!")
            #end function
            colors=None
        elif (X is None)&(self.X is not None):
            orclus = OrClus(self.X,k,l,alpha=alpha,dist=dist)
            orclus.OrClus(delta=delta)
        elif X is not None:
            orclus = OrClus(X,k,l,alpha=alpha,dist=dist)
            orclus.OrClus(delta=delta)
        
        if set_it==True:
            self.orclus = orclus.c
        else:
            return orclus.c

    #a = 0;n_c = 0
        #for i in range(k):
        #    gmm = GMM(n_components=i,n_init=10)
        #    gmm.fit(D)
        #    b = gmm.aic(Dc)*gmm.bic(Dc)
        #    a = b if b<a else a
        #    n_c = i if b<a else n_c
    #def VBGMM(self):
    #def GeneticClustering():
class ClusterViz(object):

    def PlotClusters(self,A,c,ordered=False,median=False,fig_size=(20,10),auto_min_max=False,v_min=0,v_max=300,c_bar=False,show=True):
        if type(c)==type(np.array([1])):
            x = copy.copy(c)
        else:
            x = np.array(c)
        if type(A)==type(np.array([1])):
            X = copy.copy(A)
        else:
            X = np.array(A)
        if X.shape[0]!=x.shape[0]:
            print("A and c shapes should match!!!")
        else:
            labels = list(set(x))
            n = len(labels)
            d = defaultdict()
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1,n,1)
            if auto_min_max==True:
                v_min,v_max = self.MinMaxPercentile(X)
            for i in range(n):
                d[labels[i]] = plt.subplot(1,n,i+1)
            if ordered==False:
                for l in labels:
                    sb.heatmap(X[x==l],xticklabels=False,yticklabels=False,ax=d[l],vmin=v_min,vmax=v_max,cbar=c_bar)
            else:
                for l in labels:
                    C = X[x==l]
                    if median==False:
                        u = np.mean(C,axis=0)
                    else:
                        u = np.median(C,axis=0)
                    C = np.array(sorted(C,key = lambda x: norm(x-u)))
                    sb.heatmap(C,xticklabels=False,yticklabels=False,ax=d[l],vmin=v_min,vmax=v_max,cbar=c_bar)
            
            if show==True:
                plt.show()

    def MinMax(self,A):
        if type(A)==type(np.array([1])):
            X = copy.copy(A)
        else:
            X = np.array(A)
        return np.min(X),np.max(X)
    
    def MinMaxPercentile(self,A,min=25,max=75):
        return np.percentile(A,min),np.percentile(A,max)

    def Histogram(self,A,c,dates=None,ordered=True,median=False,show=True,auto_min_max=False,percentiles=(25,75), v_min=0,v_max=300,c_bar=False,fig_size=(20,14)):
    # Histogram produces an image of the histograms of day counting for each cluster. The clustering should be inputed as the vector c.
    # the matrix of samples can be a list of lists, a numpy array or a pandas dataframe. In the first two cases, a pandas timeseries
    # with the dates of each day should be given apart. In the former case, the zeroth colummn should be provided as the vector of date
    # times.
        if c is None:
            print("c is empty, u shall fill it!!")
        else:
            if (type(A) == type(pd.DataFrame([1])))&(dates is None):
                dates = A.iloc[:,0]
                X = np.array(A.iloc[:,1:])
            elif (type(A) == type(pd.DataFrame([1])))&(dates is not None):
                X = np.array(A.iloc[:,1:])
            elif (type(A) == type(np.array([1])))&(c is not None):
                X = copy.copy(A)
            elif (type(A) != type(np.array([1])))&(type(A) != type(pd.DataFrame([1])))&(c is not None):
                X = np.array(A)
            else:
                print("should set A and c properly!")
            
            if type(c)==type(np.array([1])):
                y = copy.copy(c)
            else:
                y = np.array(c)
            
            if auto_min_max==True:
                if percentiles is not None:
                    v_min,v_max=self.MinMaxPercentile(X,min=percentiles[0],max=percentiles[1])
                else:
                    v_min,v_max=self.MinMaxPercentile(X)
            
            labels = list(set(y))
            n = len(labels)
            
            days = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}
            d = defaultdict()
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(4,np.ceil(n/2),1)

            x = np.ceil(n/2)
            for i in range(n):
                if i<x:#that's for dividing the image in two, instead of a whole block of two rows of 10 columns
                    #plotting the clusters
                    if ordered==True:
                        C = X[y==labels[i]]
                        if median==False:
                            u = np.mean(C,axis=0)
                        else:
                            u = np.median(C,axis=0)
                        C = np.array(sorted(C,key = lambda x: norm(x-u)))
                    else:
                        C = X[y==labels[i]]

                    sb.heatmap(C,xticklabels=False,yticklabels=False,ax=plt.subplot(4,x,i+1),vmin=v_min,vmax=v_max,cbar=c_bar)
                    #now, making the histogram of the week days counts! It's more complicated...
                    s = pd.Series(pd.DatetimeIndex(dates[y==labels[i]]).dayofweek).apply(lambda x: days[x]).to_frame('day')
                    s['day'] = s['day'].apply(str)
                    #next, we'll sort the days according to their counting order, 
                    #so that while graphing they will appear in ascending order
                    counts = s.day.value_counts()
                    ix = s.day.apply(lambda x: counts[x]).sort_values(ascending=True).index
                    s=s.reindex(ix)
                    sb.countplot(x='day',data=s,palette='Greens_d',ax=plt.subplot(4,x,x+i+1))
                    plt.subplot(4,x,x+i+1).set_xlabel('')
                    plt.subplot(4,x,x+i+1).set_ylabel('')
                else:
                    if ordered==True:
                        C = X[y==labels[i]]
                        if median==False:
                            u = np.mean(C,axis=0)
                        else:
                            u = np.median(C,axis=0)
                        C = np.array(sorted(C,key = lambda x: norm(x-u)))
                    else:
                        C = X[y==labels[i]]

                    sb.heatmap(C,xticklabels=False,yticklabels=False,ax=plt.subplot(4,x,x+i+1),vmin=v_min,vmax=v_max,cbar=c_bar)
                    s = pd.Series(pd.DatetimeIndex(dates[y==labels[i]]).dayofweek).apply(lambda x: days[x]).to_frame('day')
                    s['day'] = s['day'].apply(str)
                    counts = s.day.value_counts()
                    ix = s.day.apply(lambda x: counts[x]).sort_values(ascending=True).index
                    s=s.reindex(ix)
                    sb.countplot(x='day',data=s,palette='Greens_d',ax=plt.subplot(4,x,2*x+i+1))
                    plt.subplot(4,x,np.floor(n/2)+x+i+1).set_xlabel('')
                    plt.subplot(4,x,np.floor(n/2)+x+i+1).set_ylabel('')
            plt.subplot(4,x,3).set_xlabel('aaaaaaaaahhhh!!!')
            plt.subplot(4,x,1).set_ylabel('Cluster Days') 
            plt.subplot(4,x,x+1).set_ylabel('Counts')
            plt.subplot(4,x,2*x+1).set_ylabel('Cluster Days')
            plt.subplot(4,x,3*x+1).set_ylabel('Counts')
            if n%2==1:
                plt.subplot(4,x,3*x).axis('off')
                plt.subplot(4,x,4*x).axis('off')
                
            if show==True:
                plt.show()

    def RandomAdjustedHistogram(self,A,c,dates=None,sorting='min',ordered=True,median=False,show=True,auto_min_max=False,percentiles=(25,75), v_min=0,v_max=300,c_bar=False,fig_size=(20,14)):
    # RandonAdjustedHistogram produces an image of the histograms of day counting for each cluster. The clustering should be inputed as the vector c.
    # the matrix of samples can be a list of lists, a numpy array or a pandas dataframe. In the first two cases, a pandas timeseries
    # with the dates of each day should be given apart. In the former case, the zeroth colummn should be provided as the vector of date
    # times.
    # RandomAdjustedHistogram works equal Histogram, but here it adjusts the sizes of clusters view by random selecting a fixed number of them
        if c is None:
            print("c is empty, u shall fill it!!")
        else:
            if (type(A) == type(pd.DataFrame([1])))&(dates is None):
                dates = A.iloc[:,0]
                X = np.array(A.iloc[:,1:])
            elif (type(A) == type(pd.DataFrame([1])))&(dates is not None):
                X = np.array(A.iloc[:,1:])
            elif (type(A) == type(np.array([1])))&(c is not None):
                X = copy.copy(A)
            elif (type(A) != type(np.array([1])))&(type(A) != type(pd.DataFrame([1])))&(c is not None):
                X = np.array(A)
            else:
                print("should set A and c properly!")
            
            if type(c)==type(np.array([1])):
                y = copy.copy(c)
            else:
                y = np.array(c)
            
            if auto_min_max==True:
                if percentiles is not None:
                    v_min,v_max=self.MinMaxPercentile(X,min=percentiles[0],max=percentiles[1])
                else:
                    v_min,v_max=self.MinMaxPercentile(X)
            
            labels = list(set(y))
            n = len(labels)
            # Now, we define the size of each cluster, the default is to get th minimum
            lengths = [X[y==labels[i]].shape[0] for i in range(n)] #for later!
            if sorting=='min':
                size = min(lengths)
            
            days = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}
            d = defaultdict()
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(4,np.ceil(n/2),1)

            x = np.ceil(n/2)
            for i in range(n):
                if i<x:#that's for dividing the image in two, instead of a whole block of two rows of 10 columns
                    #plotting the clusters
                    C = X[y==labels[i]]
                    indexes = random.sample(range(C.shape[0]),size)
                    C = C[indexes]                    

                    if ordered==True:
                        if median==False:
                            u = np.mean(C,axis=0)
                        else:
                            u = np.median(C,axis=0)
                        C = np.array(sorted(C,key = lambda x: norm(x-u)))

                    sb.heatmap(C,xticklabels=False,yticklabels=False,ax=plt.subplot(4,x,i+1),vmin=v_min,vmax=v_max,cbar=c_bar)
                    #now, making the histogram of the week days counts! It's more complicated...
                    s = pd.Series(pd.DatetimeIndex(dates[y==labels[i]]).dayofweek).apply(lambda x: days[x]).to_frame('day')
                    s['day'] = s['day'].apply(str)
                    #next, we'll sort the days according to their counting order, 
                    #so that while graphing they will appear in ascending order
                    counts = s.day.value_counts()
                    ix = s.day.apply(lambda x: counts[x]).sort_values(ascending=True).index
                    s=s.reindex(ix)
                    sb.countplot(x='day',data=s,palette='Greens_d',ax=plt.subplot(4,x,x+i+1))
                    plt.subplot(4,x,x+i+1).set_xlabel('')
                    plt.subplot(4,x,x+i+1).set_ylabel('')
                else:
                    C = X[y==labels[i]]
                    indexes = random.sample(range(C.shape[0]),size)
                    C = C[indexes]                    

                    if ordered==True:
                        if median==False:
                            u = np.mean(C,axis=0)
                        else:
                            u = np.median(C,axis=0)
                        C = np.array(sorted(C,key = lambda x: norm(x-u)))

                    sb.heatmap(C,xticklabels=False,yticklabels=False,ax=plt.subplot(4,x,x+i+1),vmin=v_min,vmax=v_max,cbar=c_bar)
                    s = pd.Series(pd.DatetimeIndex(dates[y==labels[i]]).dayofweek).apply(lambda x: days[x]).to_frame('day')
                    s['day'] = s['day'].apply(str)
                    counts = s.day.value_counts()
                    ix = s.day.apply(lambda x: counts[x]).sort_values(ascending=True).index
                    s=s.reindex(ix)
                    sb.countplot(x='day',data=s,palette='Greens_d',ax=plt.subplot(4,x,2*x+i+1))
                    plt.subplot(4,x,np.floor(n/2)+x+i+1).set_xlabel('')
                    plt.subplot(4,x,np.floor(n/2)+x+i+1).set_ylabel('')
            plt.subplot(4,x,3).set_xlabel('aaaaaaaaahhhh!!!')
            plt.subplot(4,x,1).set_ylabel('Cluster Days') 
            plt.subplot(4,x,x+1).set_ylabel('Counts')
            plt.subplot(4,x,2*x+1).set_ylabel('Cluster Days')
            plt.subplot(4,x,3*x+1).set_ylabel('Counts')
            if n%2==1:
                plt.subplot(4,x,3*x).axis('off')
                plt.subplot(4,x,4*x).axis('off')
                
            if show==True:
                plt.show()

    def PlotGenesMatrix(self,A,big=False,show=True,c_map=None,normalize=None,v_min=None,v_max=None,alpha=None,snap=True,edgecolors=None):
        if type(A) == type(pd.DataFrame([1])):
            X = np.array(A)
        elif type(A) == type(np.array([1])):
            X = copy.copy(A)
        elif (type(A) != type(np.array([1])))&(type(A) != type(pd.DataFrame([1]))):
            X = np.array(A)
    
        x = np.arange(X.shape[0]+1)
        y = np.arange(X.shape[1]+1)
        if big==False:
            plt.pcolor(x,y,X.T,cmap=c_map,norm=normalize,vmin=v_min,vmax=v_max,alpha=alpha,snap=snap,edgecolors=edgecolors)
        else:
            plt.pcolormesh(x,y,X.T,cmap=c_map,norm=normalize,vmin=v_min,vmax=v_max,alpha=alpha,snap=snap,edgecolors=edgecolors)
        if show==True:
            plt.show()
    
    def PlotManyMatrices(self,A,c,ordered=True,median=False,show=True,auto_min_max=False,percentiles=(25,75),v_min=0,v_max=300,c_bar=False,fig_size=(20,14)):
        if c is None:
            print("c is empty, u shall fill it!!")
        else:
            if type(A) == type(pd.DataFrame([1])):
                X = np.array(A.iloc[:,1:])
            elif type(A) == type(np.array([1])):
                X = copy.copy(A)
            elif (type(A) != type(np.array([1])))&(type(A) != type(pd.DataFrame([1]))):
                try:
                    X = np.array(A)
                except:
                    print('X hasn\'t an admissible format!')
            else:
                print("should set A and c properly!")
            
            if type(c)==type(np.array([1])):
                y = copy.copy(c)
            else:
                try:
                    y = np.array(c)
                except:
                    print('c hasn\'t an admissible format!')
            
            if auto_min_max==True:
                if percentiles is not None:
                    v_min,v_max=self.MinMaxPercentile(X,min=percentiles[0],max=percentiles[1])
                else:
                    v_min,v_max=self.MinMaxPercentile(X)
            
            labels = list(set(y))
            n = len(labels)
            
            d = defaultdict()
            m = np.ceil(n/2)
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(2,m,1)

            for i in range(n):
                
                if ordered==True:
                    C = X[y==labels[i]]
                    if median==False:
                        u = np.mean(C,axis=0)
                    else:
                        u = np.median(C,axis=0)
                    C = np.array(sorted(C,key = lambda x: norm(x-u)))
                else:
                    C = X[y==labels[i]]                

                a = np.arange(C.shape[0]+1)
                b = np.arange(C.shape[1]+1)
                    
                plt.subplot(2,m,i+1).pcolormesh(a,b,C.T,cmap='jet')

            if n%2==1:
                plt.subplot(2,m,n+1).axis('off')
            if show==True:
                plt.show()
    
    def LoadPandasDataFrames(self,path,heads,delimter=';'):
    # load the data specified by the heads, which is a list of strings
    
        df = defaultdict()
        
        for i in range(len(heads)):
            p = os.path.joint(path,heads[i])
            df[i] = pd.read_csv(p,delimiter=delimiter)
        
        return df

    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    import copy
    import os
    from collections import defaultdict
    from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
    from sklearn.mixture import GMM,DPGMM
    
    path = r'C:\Users\hghghghghg\Desktop\Mestrado\PNCT-codes\Tables'
    p = 'count_eq173.csv'
    p1 = 'count_eq174.csv'
    p2 = 'count_eq182.csv'
    p3 = 'count_eq184.csv'
    p4 = 'count_eq186.csv'
    #path = r'C:\Users\hghghghghg\Desktop'
    #p = 'freq_eq_174_sent_C_sgp.csv'

    df = pd.read_csv(os.path.join(path,p),delimiter=';')
    df1 = pd.read_csv(os.path.join(path,p1),delimiter=';')
    df2 = pd.read_csv(os.path.join(path,p2),delimiter=';')
    df3 = pd.read_csv(os.path.join(path,p3),delimiter=';')
    df4 = pd.read_csv(os.path.join(path,p4),delimiter=';')
    df[:2]

    col_0 = df.columns[0]
    df = df.drop(col_0,axis=1)
    df1 = df1.drop(col_0,axis=1)
    df2 = df2.drop(col_0,axis=1)
    df3 = df3.drop(col_0,axis=1)
    df4 = df4.drop(col_0,axis=1)
    df[:2]

class SortClass(object):

    def __init__(self,l,n=1e6):
        # l = [('a',P1),('b',P2),('c',P3),('d',P4)]
        self.n = int(n)
        """o = (p[1] for p in l)
        s = 0
        z = 0
        for p in o:
            s+=np.floor(p)
            z+=np.ceil(p)
        if s == 1 & z == 1:
            self.n = 10
        else:"""
        s = 0
        for x in l:
            s += x[1]
        for i in range(len(l)):
            l[i] = l[i][0],l[i][1]/s
        self.l = l
        self.freq_list = []
        for x in self.l:
            self.freq_list += [x[0]]*int(x[1]*self.n)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self.freq_list[np.random.randint(0,len(self.freq_list))]
        #return np.random.choice(self.freq_list)
     
####################################################################################################################################
####################################################################################################################################

class OrClus(object):
# self.c ; self.E ; self.S ; self.alpha ; self.delta
    def __init__(self,X,k,l,alpha=0.7,dist='Euclid'):
    # If dist!='Euclid', then it should be given as dist = lambda x,y : dist_function(x - y) or something similar (dist = dist(x,y))
        self.k = k
        self.l = l
        self.alpha=alpha
        self.dist = dist
        
        if type(X)==type(np.array([1])):
            self.X = copy.copy(X)
        elif type(X)==type(pd.DataFrame([1])):
            self.X = np.array(X)
        elif (type(X)!=type(np.array([1])))&(type(X)!=type(pd.DataFrame([1]))):
            try:
                self.X = np.array(X)
            except:
                print("X should be in a proper format!!")

    def OrClus(self,X=None,k=None,l=None,delta=10):
        if (X==None)&(k==None)&(l==None):
            # delta should be > 1! Ensuring that k0 > k.
            self.k0 = int(np.ceil(delta*self.k))
            self.l0 = len(self.X[0])
            a = self.alpha
            b = math.log(self.l0/self.l)*math.log(1/a)/math.log(self.k0/self.k)
            b = math.exp(-b)
            n = self.X.shape[0]
            sample = random.sample(range(n),self.k0)
            
            self.S = dict()
            self.E = dict()
            for i in range(len(sample)):
                self.S[i+1] = self.X[sample[i]]
                self.E[i+1] = np.eye(self.l0)
            
            N=1
            while self.k0 > self.k:
                self.Assign(N=N,sample=sample)
                N=None
                sample=None
                
                for i in range(self.k0):
                    C = self.X[self.c==i+1]
                    if C.shape[0]==1:
                        self.E[i+1] = self.E[i+1]
                    elif C.shape[0]>1:
                        self.E[i+1] = self.FindVectors(C,self.l0)
                    else:
                        print("Empty Cluster!")

                k_new = max(self.k,int(np.floor(a*self.k0)))
                l_new = max(self.l,int(np.floor(b*self.l0)))
                print(a,b,self.l0,self.k0,self.k,k_new,self.l,int(np.floor(b*self.l0)),l_new)
                self.Merge(k_new,l_new)
                self.l0 = l_new
                print(self.E[1].shape,self.E[1])
            
            self.Assign()

    def Merge(self,k_new,l_new):
        n = len(self.c)
        while self.k0 > k_new:
            C = self.X[(self.c==1)|(self.c==2)]
            E = self.FindVectors(C,l_new)
            print(E.shape)
            C -= C.mean(axis=0)
            s = np.linalg.norm(E.dot(C.T),axis=0).mean()
            t = (1,2)
            for i,j in itertools.combinations(range(self.k0),2):
                C = self.X[(self.c==i+1)|(self.c==j+1)]                                   
                E = self.FindVectors(C,l_new)
                C -= C.mean(axis=0)
                r = np.linalg.norm(E.dot(C.T),axis=0).mean()
                t = (i+1,j+1) if r < s else t
                s = r if r < s else s
            
            self.c[self.c==t[1]]=t[0]
            self.c[self.c>t[1]] -= 1
            
            for j in range(t[1],self.k0):
                self.S[j] = self.S[j+1]
                self.E[j] = self.E[j+1]
            del self.E[self.k0],self.S[self.k0]
            
            
            self.E[t[0]] = self.FindVectors(self.X[self.c==t[0]],l_new)
            self.S[t[0]] = np.mean(self.X[self.c==t[0]],axis=0)
            
            self.k0 -= 1
    
    def FindVectors(self,C,l):
    # FindVectors returns a matrix E whose rows are the l least eigenvalued eigenvectors of the covariance matrix 
    # of the Cluster inputed (as another matrix)
        U,S,V = np.linalg.svd(np.cov(C.T))#scipy.sparse.linalg.svds(np.cov(C.T))
        return V[-l:]
        
    def Assign(self,N=None,sample=None):
    # self.S = {1:s_1,2:s_2,3:s_3, ... } self.E = {1:E_1,2E_2,3:E_3, ... }
    # where s_i the i-th centroid and E_i is the i-th matrix whose rows are eigenvectors of each cluster
    # If dist!='Euclid', then it should be given as dist = lambda x,y : dist_function(x - y) or something similar (dist = dist(x,y))
        if N==1:
            if self.dist=='Euclid':
                dist = np.linalg.norm
                colors = []
                for i in range(self.X.shape[0]):
                    if i in sample:
                        color = sample.index(i) + 1
                    else:
                        z = np.array([ dist(self.E[j].dot(self.X[i] - self.S[j])) for j in iter(self.S)])
                        z = z[~np.isnan(z)]
                        color = np.argmin(z)+1
                    colors.append(color)
                    
                self.c = np.array(colors)
                for i in iter(self.S):
                    A = self.X[self.c==i]
                    self.S[i] = np.mean(A,axis=0)
                
            else:
                colors = []
                for x in X:
                    color = 1   #color = list(S_E.keys())[0]
                    s = self.dist(self.E[1].dot(x),self.E[1].dot(self.S[1]))
                    for i in iter(S_E):
                        distance = dist(self.E[i].dot(x),self.E[i].dot(self.S[i]))
                        color = i if (distance < s) else color
                    colors.append(color)
                self.c = np.array(colors)
                for i in iter(self.S):
                    print('this')
            
        elif N==None:
            if self.dist=='Euclid':
                dist = np.linalg.norm
                colors = []
                for x in self.X:
                    z = np.array([ dist(self.E[i].dot(x - self.S[i])) for i in iter(self.S)])
                    color = np.argmin(z)+1
                    colors.append(color)
                    self.c = np.array(colors)
                sample2 = random.sample(range(self.X.shape[0]),len(self.S))
                for i in iter(self.S):
                    A = self.X[self.c==i]
                    if A.shape[0]==0:
                        j = sample2[i-1]
                        self.S[i] = self.X[j]
                        self.c[j] = i
                    elif A.shape[0] > 0:
                        self.S[i] = np.mean(A,axis=0)
                    else:
                        print('strange thing happening, X.shape not 0 nor > 0 !')
                
            else:
                colors = []
                for x in X:
                    color = 1   
                    s = self.dist(self.E[1].dot(x),self.E[1].dot(self.S[1]))
                    for i in iter(S_E):
                        distance = dist(self.E[i].dot(x),self.E[i].dot(self.S[i]))
                        color = i if (distance < s) else color
                    colors.append(color)
                self.c = np.array(colors)
                for i in iter(self.S):
                    self.S[i] = np.mean(self.X[self.c==i],axis=0)

###########################################################################################################################################
##########################################################################################################################################

class ORCLUS(object):

    def __init__(self,X,k,l,alpha=0.7,dist='Euclid'):
    # If dist!='Euclid', then it should be given as dist = lambda x,y : dist_function(x - y) or something similar (dist = dist(x,y))
        self.k = k
        self.l = l
        self.alpha=alpha
        self.dist = dist
        
        if type(X)==type(np.array([1])):
            self.X = copy.copy(X)
        elif type(X)==type(pd.DataFrame([1])):
            self.X = np.array(X)
        elif (type(X)!=type(np.array([1])))&(type(X)!=type(pd.DataFrame([1]))):
            try:
                self.X = np.array(X)
            except:
                print("X should be in a proper format!!")

    def ORCLUS(self,X=None,k=None,l=None,delta=10):
        if (X==None)&(k==None)&(l==None):
            # delta should be > 1! Ensuring that k0 > k.
            self.k0 = int(np.ceil(delta*self.k))
            self.l0 = len(self.X[0])
            a = self.alpha
            b = math.log(self.l0/self.l)*math.log(1/a)/math.log(self.k0/self.k)
            b = math.exp(-b)
            n = self.X.shape[0]
            sample = random.sample(range(n),self.k0)
            
            self.S = dict()
            self.E = dict()
            for i in range(len(sample)):
                self.S[i+1] = self.X[sample[i]]
                self.E[i+1] = np.eye(self.l0)
            #print(self.S)
            #print(self.E)
            
            N=1
            while self.k0 > self.k:
                self.Assign(N=N,sample=sample)
                N=None
                sample=None
                #print(self.c)
                #print(self.E)
                print(self.S)
                print(self.k0,self.k,len(set(self.c)))
                
                for i in range(self.k0):
                    C = self.X[self.c==i+1]
                    if C.shape[0]==1:
                        self.E[i+1] = self.E[i+1]
                    elif C.shape[0]>1:
                        self.E[i+1] = self.FindVectors(C,self.l0)
                    else:
                        print("Empty Cluster!")
                    #try:
                    #    if (self.S[i].shape[0]!=1)&(self.S[i+1].shape[1]!=0):
                    #        self.E[i+1] = self.FindVectors(self.X[self.c==i+1],self.l0)
                    #except:
                    #    self.E[i+1] = self.E[i+1]
                print('hello')
                k_new = max(self.k,int(np.floor(a*self.k0)))
                l_new = max(self.l,int(np.floor(b*self.l0)))
                print(self.l,int(np.floor(b*self.l0)),l_new)
                self.Merge(k_new,l_new)
                print('hello')
            
            self.Assign()

    def Merge(self,k_new,l_new):
        """for i,j in itertools.combinations(range(self.k0)):
            C = np.vstack((self.X[self.c==i+1],self.X[self.c==j+1]))
            #s = C.mean(axis=0)
            E = self.FindVectors(C,l_new)
            C -= C.mean(axis=0)
            r = np.linalg.norm(E.dot(C.T),axis=0)
            r = np.mean(r)"""
        n = len(self.c)
        while self.k0 > k_new:
            #C = np.vstack((self.X[self.c==1],self.X[self.c==1]))
            C = self.X[(self.c==1)|(self.c==2)]
            #s = C.mean(axis=0)
            print('track 1')
            E = self.FindVectors(C,l_new)
            C -= C.mean(axis=0)
            s = np.linalg.norm(E.dot(C.T),axis=0).mean()
            t = (1,2)
            for i,j in itertools.combinations(range(self.k0),2):
                #C = np.vstack((self.X[self.c==i+1],self.X[self.c==j+1]))
                C = self.X[(self.c==i+1)|(self.c==j+1)]
                #s = C.mean(axis=0)                                    
                E = self.FindVectors(C,l_new)
                C -= C.mean(axis=0)
                r = np.linalg.norm(E.dot(C.T),axis=0).mean()
                t = (i+1,j+1) if r < s else t
                s = r if r < s else s
            
            self.c[self.c==t[1]]=t[0]
            self.c[self.c>t[1]] -= 1
            
            for j in range(t[1],self.k0):
                self.S[j] = self.S[j+1]
                self.E[j] = self.E[j+1]
            del self.E[self.k0],self.S[self.k0]
            
            
            self.E[t[0]] = self.FindVectors(self.X[self.c==t[0]],l_new)
            self.S[t[0]] = np.mean(self.X[self.c==t[0]],axis=0)
            
            self.k0 -= 1
    
    def FindVectors(self,C,l):
    # FindVectors returns a matrix E whose rows are the l least eigenvalued eigenvectors of the covariance matrix 
    # of the Cluster inputed (as another matrix)
        #A = copy.deepcopy(C)
        #A -= A.mean(axis=0)
        #Cov = A.T.dot(A)
        U,S,V = scipy.sparse.linalg.svds(np.cov(C.T))
        #E = copy.deepcopy(V[-l:])
        #del Cov,U,S,V
        #return E
        return V[-l:]
        
    def Assign(self,N=None,sample=None):
    # self.S = {1:s_1,2:s_2,3:s_3, ... } self.E = {1:E_1,2E_2,3:E_3, ... }
    # where s_i the i-th centroid and E_i is the i-th matrix whose rows are eigenvectors of each cluster
    # If dist!='Euclid', then it should be given as dist = lambda x,y : dist_function(x - y) or something similar (dist = dist(x,y))
        if N==1:
            if self.dist=='Euclid':
                dist = np.linalg.norm
                colors = []
                for i in range(self.X.shape[0]):
                    if i in sample:
                        color = sample.index(i) + 1
                    else:
                        z = np.array([ dist(self.E[j].dot(self.X[i] - self.S[j])) for j in iter(self.S)])
                        #print(z)
                        z = z[~np.isnan(z)]
                        color = np.argmin(z)+1
                    colors.append(color)
                    
                self.c = np.array(colors)
                for i in iter(self.S):
                    A = self.X[self.c==i]
                    #n = len(A)
                    self.S[i] = np.mean(A,axis=0)
                    #if n>1 else A.flatten()
                
            else:
                colors = []
                for x in X:
                    color = 1   #color = list(S_E.keys())[0]
                    s = self.dist(self.E[1].dot(x),self.E[1].dot(self.S[1]))
                    for i in iter(S_E):
                        distance = dist(self.E[i].dot(x),self.E[i].dot(self.S[i]))
                        color = i if (distance < s) else color
                    colors.append(color)
                self.c = np.array(colors)
                for i in iter(self.S):
                    print('this')
            
        elif N==None:
            if self.dist=='Euclid':
                dist = np.linalg.norm
                colors = []
                for x in self.X:
                    #check=False
                    #for i in iter(self.S):
                    #    if np.allclose(x,self.S[i]):
                    #        check = True
                    #        color = i
                    #        #break
                    #if check==False:
                    z = np.array([ dist(self.E[i].dot(x - self.S[i])) for i in iter(self.S)])
                    #print(z)
                    #z = z[~np.isnan(z)]
                    color = np.argmin(z)+1
                    #d = min([ dist(self.E[color].dot(self.S[color] - self.S[i])) for i in iter(self.S) ])
                    #color = -1 if d <= z[color-1] else color
                    colors.append(color)
                        
                    #color = 1
                    #s = dist(self.E[1].dot(x - self.S[1]))
                    #for i in iter(self.S):
                        #if np.allclose(x,self.S[i]):
                        #    color = i
                        #    s = -1
                        #else:
                        #    distance = dist(self.E[i].dot(x - self.S[i]))
                        #    color = i if (distance < s) else color
                        #    s = distance if (distance < s) else s
                    #colors.append(color)
                    #color = sorted([(i,dist(self.E[i].dot(x - self.S[i]))) for i in iter(self.S)],key=lambda x: x[1])[0][0]"""
                    #colors.append(color)
                #print(len(set(colors)))
                #print(colors)
                self.c = np.array(colors)
                sample2 = random.sample(range(self.X.shape[0]),len(self.S))
                for i in iter(self.S):
                    A = self.X[self.c==i]
                    if A.shape[0]==0:
                        j = sample2[i-1]#random.choice(range(self.X.shape[0]))
                        self.S[i] = self.X[j]
                        self.c[j] = i
                    elif A.shape[0] > 0:
                    #n = len(A)
                        self.S[i] = np.mean(A,axis=0) #if n>1 else A.flatten()
                    else:
                        print('strange thing happening, X.shape not 0 nor > 0 !')
                
            else:
                colors = []
                for x in X:
                    color = 1   #color = list(S_E.keys())[0]
                    s = self.dist(self.E[1].dot(x),self.E[1].dot(self.S[1]))
                    for i in iter(S_E):
                        distance = dist(self.E[i].dot(x),self.E[i].dot(self.S[i]))
                        color = i if (distance < s) else color
                    colors.append(color)
                self.c = np.array(colors)
                for i in iter(self.S):
                    self.S[i] = np.mean(self.X[self.c==i],axis=0)

############################################# ECF Optimizations! #############################################################3

    def ECFORCLUS(self,X=None,k=None,l=None,delta=10):
        if (X==None)&(k==None)&(l==None):
            # delta should be > 1! Ensuring that k0 > k.
            self.k0 = int(np.ceil(delta*self.k))
            self.l0 = len(self.X[0])
            a = self.alpha
            b = math.log(self.l0/self.l)*math.log(1/a)/math.log(self.k0/self.k)
            b = math.exp(-b)
            n = self.X.shape[0]
            sample = random.sample(range(n),self.k0)
            
            self.S = dict()
            self.E = dict()
            self.ECF = dict()
            for i in range(len(sample)):
                self.S[i+1] = self.X[sample[i]]
                self.E[i+1] = np.eye(self.l0)
            #print(self.S)
            #print(self.E)
            
            N=1
            while self.k0 > self.k:
                self.ECFAssign(N=N,sample=sample)
                N=None
                sample=None
                #print(self.c)
                #print(self.E)
                print(self.S)
                print(self.k0,self.k,len(set(self.c)))
                
                for i in range(self.k0):
                    C = self.X[self.c==i+1]
                    if C.shape[0]==1:
                        self.ECF[i+1] = np.array((C.T.dot(C),C.sum(axis=0),C.shape[0]))
                        self.E[i+1] = self.E[i+1]
                    elif C.shape[0]>1:
                        self.ECF[i+1] = np.array((C.T.dot(C),C.sum(axis=0),C.shape[0]))
                        self.E[i+1] = self.ECFFindVectors(self.ECF[i+1],self.l0)
                    else:
                        print("Empty Cluster!")
                    #try:
                    #    if (self.S[i].shape[0]!=1)&(self.S[i+1].shape[1]!=0):
                    #        self.E[i+1] = self.FindVectors(self.X[self.c==i+1],self.l0)
                    #except:
                    #    self.E[i+1] = self.E[i+1]
                print('hello')
                k_new = max(self.k,int(np.floor(a*self.k0)))
                l_new = max(self.l,int(np.floor(b*self.l0)))
                self.ECFMerge(k_new,l_new)
                print('hello')
            
            self.ECFAssign()

    def ECFFindVectors(self,ECF,l):
    # FindVectors returns a matrix E whose rows are the l least eigenvalued eigenvectors of the covariance matrix 
    # of the Cluster inputed (as another matrix)
        cov = ECF[0] - np.outer(ECF[1],ECF[1])
        U,S,V = scipy.sparse.linalg.svds(cov/(ECF[2]**2))
        return V[-l:]

    def ECFAssign(self,N=None,sample=None):
    # self.S = {1:s_1,2:s_2,3:s_3, ... } self.E = {1:E_1,2E_2,3:E_3, ... }
    # where s_i the i-th centroid and E_i is the i-th matrix whose rows are eigenvectors of each cluster
    # If dist!='Euclid', then it should be given as dist = lambda x,y : dist_function(x - y) or something similar (dist = dist(x,y))
        if N==1:
            if self.dist=='Euclid':
                dist = np.linalg.norm
                colors = []
                for i in range(self.X.shape[0]):
                    if i in sample:
                        color = sample.index(i) + 1
                    else:
                        z = np.array([ dist(self.E[j].dot(self.X[i] - self.S[j])) for j in iter(self.S)])
                        #print(z)
                        z = z[~np.isnan(z)]
                        color = np.argmin(z)+1
                    colors.append(color)
                    
                self.c = np.array(colors)
                for i in iter(self.S):
                    A = self.X[self.c==i]
                    #n = len(A)
                    self.S[i] = np.mean(A,axis=0)
                    #if n>1 else A.flatten()
                
            else:
                colors = []
                for x in X:
                    color = 1   #color = list(S_E.keys())[0]
                    s = self.dist(self.E[1].dot(x),self.E[1].dot(self.S[1]))
                    for i in iter(S_E):
                        distance = dist(self.E[i].dot(x),self.E[i].dot(self.S[i]))
                        color = i if (distance < s) else color
                    colors.append(color)
                self.c = np.array(colors)
                for i in iter(self.S):
                    print('this')
            
        elif N==None:
            if self.dist=='Euclid':
                dist = np.linalg.norm
                colors = []
                for x in self.X:
                    #check=False
                    #for i in iter(self.S):
                    #    if np.allclose(x,self.S[i]):
                    #        check = True
                    #        color = i
                    #        #break
                    #if check==False:
                    z = np.array([ dist(self.E[i].dot(x - self.S[i])) for i in iter(self.S)])
                    #print(z)
                    #z = z[~np.isnan(z)]
                    color = np.argmin(z)+1
                    #d = min([ dist(self.E[color].dot(self.S[color] - self.S[i])) for i in iter(self.S) ])
                    #color = -1 if d <= z[color-1] else color
                    colors.append(color)
                        
                    #color = 1
                    #s = dist(self.E[1].dot(x - self.S[1]))
                    #for i in iter(self.S):
                        #if np.allclose(x,self.S[i]):
                        #    color = i
                        #    s = -1
                        #else:
                        #    distance = dist(self.E[i].dot(x - self.S[i]))
                        #    color = i if (distance < s) else color
                        #    s = distance if (distance < s) else s
                    #colors.append(color)
                    #color = sorted([(i,dist(self.E[i].dot(x - self.S[i]))) for i in iter(self.S)],key=lambda x: x[1])[0][0]"""
                    #colors.append(color)
                #print(len(set(colors)))
                #print(colors)
                self.c = np.array(colors)
                sample2 = random.sample(range(self.X.shape[0]),len(self.S))
                for i in iter(self.S):
                    A = self.X[self.c==i]
                    if A.shape[0]==0:
                        j = sample2[i-1]#random.choice(range(self.X.shape[0]))
                        self.S[i] = self.X[j]
                        self.c[j] = i
                    elif A.shape[0] > 0:
                    #n = len(A)
                        self.S[i] = np.mean(A,axis=0) #if n>1 else A.flatten()
                    else:
                        print('strange thing happening, X.shape not 0 nor > 0 !')
                
            else:
                colors = []
                for x in X:
                    color = 1   #color = list(S_E.keys())[0]
                    s = self.dist(self.E[1].dot(x),self.E[1].dot(self.S[1]))
                    for i in iter(S_E):
                        distance = dist(self.E[i].dot(x),self.E[i].dot(self.S[i]))
                        color = i if (distance < s) else color
                    colors.append(color)
                self.c = np.array(colors)
                for i in iter(self.S):
                    self.S[i] = np.mean(self.X[self.c==i],axis=0)

    def ECFMerge(self,k_new,l_new):
        n = len(self.c)
        while self.k0 > k_new:
            #C = np.vstack((self.X[self.c==1],self.X[self.c==1]))
            C = self.X[(self.c==1)|(self.c==2)]
            #s = C.mean(axis=0)
            C -= C.mean(axis=0)
            print('track 1')
            E = self.ECFFindVectors(self.ECF[1] + self.ECF[2],l_new)
            s = np.linalg.norm(E.dot(C.T),axis=0).mean()
            t = (1,2)
            for i,j in itertools.combinations(range(self.k0),2):                                    
                E = self.ECFFindVectors(self.ECF[i+1]+self.ECF[j+1],l_new)
                C -= C.mean(axis=0)
                r = np.linalg.norm(E.dot(C.T),axis=0).mean()
                t = (i+1,j+1) if r < s else t
                s = r if r < s else s
            
            self.c[self.c==t[1]]=t[0]
            self.c[self.c>t[1]] -= 1
            
            self.ECF[t[0]] += self.ECF[t[1]]
            for j in range(t[1],self.k0):
                self.S[j] = self.S[j+1]
                self.E[j] = self.E[j+1]
                self.ECF[j] = self.ECF[j+1]
            del self.E[self.k0],self.S[self.k0],self.ECF[self.k0]
 
            self.E[t[0]] = self.ECFFindVectors(self.ECF[t[0]],l_new)
            self.S[t[0]] = self.ECFnp.mean(self.X[self.c==t[0]],axis=0)
            
            self.k0 -= 1

    def ECFVector(self,C):
        ecf1 = C.T.dot(C) #.flatten()
        ecf2 = C.sum(axis=0)
        ecf3 = C.shape[0]
        return np.array((ecf1,ecf2,ecf3))

#################################################################################################################################
####################################################################################################################################

def ScatterMatrix(X,tss=False):
    
    if type(X) != type(np.array([1])):
        A = np.array(X)
    else:
        A = copy.copy(X)
        
    A = A - np.mean(A,axis=0)
    A = A.T.dot(A)
    if tss==True:
        return np.trace(A)
    else:
        return A

def PlotClusters(colors, X):
### X = [x_1,x_2,x_3, ... x_n]; x_i = [x_i_1,x_i_2, ... x_i_d]; (X can be in numpy format)
### colors = [l_1, l_2, ... l_n]; l_i \in {1,2,3,4 ..., k};
### X = [samples x features] matrix ( X is an n x d matrix, whose i-th row x_i is one among the n pieces of data, and has d features)
### colors = [1 x n] matrix in which each value of the i-th column is the Cluster number (between 1 and k) to which the 
### i-th datapoint x_i belongs to.


    Y = [list(x) for x in X]
    Y = [tuple(colors)] + list(zip(*Y))
    Xc = [list(y) for y in zip(*Y)]
    Xc = [len(set(colors))] + Xc


    PlotColoredCluster(Xc)

def PlotColoredCluster(Xc):
    X = Xc[1:]
    cs = [tuple(np.random.rand(3)) for k in range(Xc[0])]
    colors = [cs[x[0]] for x in X]
    X = [list(x[1:]) for x in X]
    plt.scatter(*zip(*X),color=colors)
    plt.show(block=False)

############################################################################################################################################################
################# Some functions borrowed from KMeans.py and PNCT.py #######################################################################################
############################################################################################################################################################

def CreateMatrixOfValues(path):
    A = []
    with open(path) as f:
        reader = csv.reader(f)
        for r in reader:
            A.append([float(s) for s in r[0].split(';')[2:]])
            
    return np.array(A)

def PlotGenesMatrix(M):
    x = np.arange(M.shape[0])
    y = np.arange(M.shape[1])
    plt.pcolor(x,y,np.transpose(M))
    plt.show()

############################################################################################################################################################
############################ GaussianMix Source code, for easy code handling here! #########################################################################
############################################################################################################################################################
############################ See below in the end for crtl c + ctrl v piece of code to generate nice testing sets ##########################################
############################################################################################################################################################

def Sort(l,n=1e4):
    # l = [('a',P1),('b',P2),('c',P3),('d',P4)]
    s = 0
    for x in l:
        s += x[1]
    for x in l:
        x = (x[0],x[1]/s)
    o = (np.floor(p[1]) for p in l)
    s = 0
    for p in o:
        s+=p
    if s == 1:
        n = 10
    
    freq_list = []
    for x in l:
        freq_list += [x[0]]*int(x[1]*n)
    freq_list = np.array(freq_list)
    return np.random.choice(freq_list)

def SortGenerator(l,n=1e6):
        s = 0
        for x in l:
            s += x[1]
        for k in range(len(l)):
            l[k] = (l[k][0],l[k][1]/s)
        """o = (np.floor(p[1]) for p in l)
        s = 0
        for p in o:
            s+=p
        if s == 1:
            n = 10"""
        freq_list = []
        for x in l:
            freq_list += [x[0]]*int(x[1]*n)
        while True:
            m = len(freq_list)
            yield freq_list[np.random.randint(0,m)]

def GaussianMix(p,m,N=1e6):
    # p = [P0,P1,P2,P3,P4]  Pk - relative proportions of gaussian mixture, to be yet normalized
    # m = [(mu0,sigma0),(mu1,sigma1),(mu2,sigma2),(mu3,sigma3),(mu4,sigma4)]
    # l = [(0,P0),(1,P1),(2,P2),(3,P3),(4,P4)]
    l = [(i,p[i]) for i in range(len(p))]
    s = SortGenerator(l,n=N)
    t = type(np.array([]))
    if all((type(x[0]),type(x[1])) == (t,t) for x in m)==True:
        dim = len(m[0][0])
        while True:
            k = next(s)
            yield m[k][0] + m[k][1].dot(np.random.randn(dim))
    else:
        print("mus and sigmas in m must be numpy arrays!")

class SortClass(object):

    def __init__(self,l,n=1e6):
        # l = [('a',P1),('b',P2),('c',P3),('d',P4)]
        self.n = int(n)
        """o = (p[1] for p in l)
        s = 0
        z = 0
        for p in o:
            s+=np.floor(p)
            z+=np.ceil(p)
        if s == 1 & z == 1:
            self.n = 10
        else:"""
        s = 0
        for x in l:
            s += x[1]
        for i in range(len(l)):
            l[i] = l[i][0],l[i][1]/s
        self.l = l
        self.freq_list = []
        for x in self.l:
            self.freq_list += [x[0]]*int(x[1]*self.n)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self.freq_list[np.random.randint(0,len(self.freq_list))]
        #return np.random.choice(self.freq_list)

def RotationMatrix(phi,dim=2):
    if dim == 2:
        phi = [phi]
        R = [[np.cos(phi[0]),np.sin(phi[0])],[-np.sin(phi[0]),np.cos(phi[0])]]
        return np.array(R)
    #elif dim == 3:

def TonsOfIt(p,m,num,N=1e6):
    g = GaussianMix(p,m,N=N)
    l = []
    for i in range(num):
        l.append(next(g))
    return np.array(l)

def Sqrt(A):
    # A = np.array([[ ... ],[ ... ], ...])
    U,T,V = np.linalg.svd(A)
    B = U.dot((np.eye(len(T))*np.sqrt(T)).dot(V))
    return B

def Covariance(D,dim=2):
    # D = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5], ...,[xn,yn]])
    if dim == 2:
        m = np.array([np.mean(D[:,0]),np.mean(D[:,1])])
        cov = np.zeros((2,2))
        cov[0,0] = np.mean(((D-m)**2)[:,0])
        cov[1,1] = np.mean(((D-m)**2)[:,1])
        cov[0,1] = np.mean((D-m)[:,0]*(D-m)[:,1])
        cov[1,0] = cov[0,1]    
        return cov
        
path = r'C:\Users\hghghghghg\Desktop\Mestrado\PNCT-codes\Tables'
p = 'count_eq173.csv'
p1 = 'count_eq174.csv'
p2 = 'count_eq182.csv'
p3 = 'count_eq184.csv'
p4 = 'count_eq186.csv'
#path = r'C:\Users\hghghghghg\Desktop'
#p = 'freq_eq_174_sent_C_sgp.csv'

df = pd.read_csv(os.path.join(path,p),delimiter=';')
df1 = pd.read_csv(os.path.join(path,p1),delimiter=';')
df2 = pd.read_csv(os.path.join(path,p2),delimiter=';')
df3 = pd.read_csv(os.path.join(path,p3),delimiter=';')
df4 = pd.read_csv(os.path.join(path,p4),delimiter=';')

# p = [P0,P1,P2,P3,P4]  Pk - relative proportions of gaussian mixture, to be yet normalized
# m = [(mu0,sigma0),(mu1,sigma1),(mu2,sigma2),(mu3,sigma3),(mu4,sigma4)]
# l = [(0,P0),(1,P1),(2,P2),(3,P3),(4,P4)]
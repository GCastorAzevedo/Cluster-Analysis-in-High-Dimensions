class OrClus(object):
# self.c ; self.E ; self.S ; self.alpha ; self.delta
	def __init__(self,X,k,l,alpha=0.7):
		# X = n_samples x n_features numpy array,    
		# X = [x1,x2,x3, ... ] , xi = [xi1,xi2, ... ]
		self.k = k
		self.l = l
		self.alpha=alpha
		self.dist = dist

	def OrClus(self,X=None,k=None,l=None,delta=10):

	################## main ##########################     

		if (X==None)&(k==None)&(l==None):
			# delta should be > 1! Ensuring that k0 > k.
			# initial k0 is delta times bigger than k   
			self.k0 = int(np.ceil(delta*self.k))
			# initial l0 is equal to full dimension d
			self.l0 = len(self.X[0])
			# alpha
			a = self.alpha
			# calculates beta
			b = math.log(self.l0/self.l)*math.log(1/a)
			b/=math.log(self.k0/self.k)
			b = math.exp(-b)
			n = self.X.shape[0]
			# selects k0 random centroids among the data set X
			sample = random.sample(range(n),self.k0)

			# start with the selected centroids and identity matrices
			self.S = dict()
			self.E = dict()
			for i in range(len(sample)):
				self.S[i+1] = self.X[sample[i]]
				self.E[i+1] = np.eye(self.l0)
				
			# starts while loop
			while self.k0 > self.k:
				# skip further to assign method for comments
				self.Assign()

				#calculates the projected space using FindVectors method
				for i in range(self.k0):
					C = self.X[self.c==i+1]
					# checks if there are more than one vectors     
					# (else we cannot calculate a covariance matrix)
					if C.shape[0]==1:
						self.E[i+1] = self.E[i+1]
					elif C.shape[0]>1:
						self.E[i+1] = self.FindVectors(C,self.l0)
					else:
						print("Empty Cluster!")

				# updates k (k(t)) to k_new (k(t+1))
				k_new = max(self.k,int(np.floor(a*self.k0)))
				# updates l (l(t)) to l_new (l(t+1))
				l_new = max(self.l,int(np.floor(b*self.l0)))
				# merges the clusters which minimizes the most the
				# 'projected energy' cost function, until k and l 
				# reduce to k_new and l_new.                      
				self.Merge(k_new,l_new)
				self.l0 = l_new

			# finally the last assignment
			self.Assign()

	############# end main ###########################	                                                                      
	############# procedures #########################

	def Merge(self,k_new,l_new):
		n = len(self.c)
		while self.k0 > k_new:
			# starts merging the first two clusters 
			# to initialize the loop minimization of
			# the 'projected energy' r
			C = self.X[(self.c==1)|(self.c==2)]
			# recalculates the projected space
			E = self.FindVectors(C,l_new)
			# calclates the associated 'projected energy' cost function
			C -= C.mean(axis=0)
			s = np.linalg.norm(E.dot(C.T),axis=0).mean()
			t = (1,2)
			# starts merging iteration process. This piece of code      
			# is highly paralellizable, as each such iteration in       
			# the FOR loop can be calculated separately (only the       
			# minimization of r should be performed  quickly thereafter)
			for i,j in itertools.combinations(range(self.k0),2):
				C = self.X[(self.c==i+1)|(self.c==j+1)]                                   
				E = self.FindVectors(C,l_new)
				C -= C.mean(axis=0)
				# projected energy
				r = np.linalg.norm(E.dot(C.T),axis=0).mean()
				# checks if the prejected energy gets lower,
				# updates the minimum index in this case    
				t = (i+1,j+1) if r < s else t
				s = r if r < s else s
			
			# updates the clusters' labels to the last      
			# occurrence of t = (i_min,j_min), corresponding
			# to r_min. because we exclude the j_min - th   
			self.c[self.c==t[1]]=t[0]
			self.c[self.c>t[1]] -= 1
			
			# updates centroids (excludes the j_min - th)
			# updates the projected spaces               
			for j in range(t[1],self.k0):
				self.S[j] = self.S[j+1]
				self.E[j] = self.E[j+1]
			del self.E[self.k0],self.S[self.k0]
			
			# recalculates the projected space and centroid       
			# of the newly merged cluster C_i' = C_i_min U C_j_min
			self.E[t[0]] = self.FindVectors(self.X[self.c==t[0]],l_new)
			self.S[t[0]] = np.mean(self.X[self.c==t[0]],axis=0)
			
			self.k0 -= 1

	def FindVectors(self,C,l):
	# FindVectors returns a matrix E whose rows are the l least
	# eigenvalued eigenvectors of the covariance matrix        
	# of the Cluster inputed (as another matrix)               
		U,S,V = np.linalg.svd(np.cov(C.T))
				#scipy.sparse.linalg.svds(np.cov(C.T))
		return V[-l:]
		
def Assign(self):
# self.S = {1:s_1,2:s_2,3:s_3, ... }                            
# self.E = {1:E_1,2E_2,3:E_3, ... }                             
# where s_i the i-th centroid and E_i is the i-th matrix whose  
# rows are eigenvectors of each cluster If dist!='Euclid', then 
# it should be given as dist = lambda x,y : dist_function(x - y)
# or something similar (dist = dist(x,y))                       

	dist = np.linalg.norm
	colors = []
	for x in self.X:
		z = np.array([ dist(self.E[i].dot(x - self.S[i])) for i in iter(self.S)])
		# color = cluster labels vector
		color = np.argmin(z)+1
		colors.append(color)
		self.c = np.array(colors)
	# After the first iteration, it starts to appear sometimes           
	# empty clusters. sample2 will be needed if they happen to appear
	sample2 = random.sample(range(self.X.shape[0]),len(self.S))
	for i in iter(self.S):
		A = self.X[self.c==i]
		# when it happens to appear empty clusters, we force 
		# them not to be so by randomly choosing one centroid
	if A.shape[0]==0:
		j = sample2[i-1]
		self.S[i] = self.X[j]
		self.c[j] = i
	elif A.shape[0] > 0:
		self.S[i] = np.mean(A,axis=0)
	else:
		print('strange thing happening, X.shape not 0 nor > 0 !')

	################ end procedures ############

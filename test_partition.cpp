/*! Adam Craig
 *  Summer 2019 Internship under Prof. Vassilevski
 *  Portland State University
	
 *  This file will test several different preconditioners
 *  for the conjugate gradient algorithm, with an emphasis on
 *  sparse matrices and parallel preconditioners
*/

#include <random>
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <metis.h>
#include "partition.hpp"

using namespace linalgcpp;

// input x, y are integer coordinates in [m]
// in the context of an m-by-m matrix
// input n = sqrt(m)
// output SPD entry A_(x,y)
double kernel(int x, int y, int n);

// input A SPD sparse matrix
// input x a random vector to create an image b
// which will be solved by a parallel CG algo
void ParCGtest(SparseMatrix<double>& A, Vector<double>& x);


// input A SPD sparse matrix
// input x a random vector to create an image b
// which will be solved by a sequential CG algo
void CGtest(SparseMatrix<double>& A, Vector<double>& x);

// input A SPD matrix
// input x random vector
// input M a preconditioner
// input (*precond) is a function that returns the
// inverse action of the preconditioner M
template <typename T>
void PCG_(SparseMatrix<T>& A, Vector<double>& x, SparseMatrix<T>& M, Vector<double> (*precond)(SparseMatrix<T>& M, Vector<double>& r));

// input M Jacobian preconditioner
// input r image to solve for (Mz = r)
// return value is z
template <typename T>
Vector<double> JacobianInv(SparseMatrix<T>& M, Vector<double>& r);


// input M Gauss-Seidel preconditioner
// input r image to solve for (Mz = r)
// return value is z
template <typename T>
Vector<double> GaussSInv(SparseMatrix<T>& M, Vector<double>& r);

// input M is SPD matrix A (preconditioner generated randomly at each iteration)
// input r image for solve for (Mz = r)
// return value is z
template <typename T>
Vector<double> TLInv(SparseMatrix<T>& M, Vector<double>& r);


// input M is SPD matrix A (preconditioner generated randomly at each iteration)
// input r image for solve for (Mz = r)
// return value is z
template <typename T>
Vector<double> MLInv(SparseMatrix<T>& M, Vector<double>& r);

// input L is the matrix which will store the (modified) SPD Laplacian
// input Adjacency is the matrix which will store the Adjacency matrix
template <typename T>
void EdgeRead(SparseMatrix<T>& Laplacian, SparseMatrix<T>& Adjacency);

// input A is an adjaency matrix
// output is an SPD Laplacian
template<typename T>
SparseMatrix<T> ModLaplacian(SparseMatrix<T>& A);

// input A is a matrix
// input row is a row
// return value is number of non-zero entries in that row
template<typename T> 
int RowSum(SparseMatrix<T>& A, int row);

// this function produces P in parallel
// input Weights is a weighted SPD Laplacian matrix
// return value is a interpolator matrix P
SparseMatrix<int> PLubysInterpolator(SparseMatrix<double>& Weights);


// this function produces P sequentially
// input Weights is a weighted SPD Laplacian matrix
// return value is a interpolator matrix P
SparseMatrix<int> LubysInterpolator(SparseMatrix<double>& Weights);

// the function finds the value of the highest entry 
// in a single row
// input Weights a weighted SPD Laplacian 
// input row the row to analyze
// return value is highest entry
double MaxRow(SparseMatrix<double>& Weights, int row);

// input A is a SPD Matrix
// return value a matrix with random (symmetric) weights
template <typename T>
SparseMatrix<double> ProduceWeights(SparseMatrix<T>& A);

// parallel sparse-matrix-vector multiplication
// input A a sparse matrix
// input x any vector
// return value A*x
template <typename T>
Vector<double> SVM(SparseMatrix<T>& A, Vector<double>& x);

// inverse action of the Jacobian preconditioner
// input A the Jacobian is a matrix
// input x the image to solve for
// return value y, Ay = x
Vector<double> JacobiInverse(SparseMatrix<double>& A, Vector<double>& x);

// inverse action of a twolevel preconditioner
// input A the SPD laplacian
// input P an interpolator matrix
// input M is A (not used)
// input r is the image to solve for
// return value is z, Mz = r
template <typename T>
Vector<double> TLNF(SparseMatrix<T>& A, SparseMatrix<int>& P, SparseMatrix<T>& M, Vector<double> r);

// inverse action of a multi-level preconditioner
// input A the SPD laplacian
// input Lmax the maximum number of recursive calls this function can make
// input Ncoarse the upper bound on the smallest system to be solved directly 
// input nk the current coarseness of the system
// input k current iteration with k<=Lmax
// input q is the factor by which nk decreases at each iteration (currently unused)
// input r is the image in the system Mz = r
// return value is z
template <typename T>
Vector<double> MLNF(SparseMatrix<T>& A, int Lmax, int Ncoarse, int nk, int k, double q, Vector<double> r);

// pre-smoother operation
// input A an SPD matrix
// input b any vector in the image of A
// input x is a any std::vector of equal size
// return value a coarse approximation of y, in Ay = b
template<typename T>
Vector<double> ForwardGaussS(SparseMatrix<T>& A, Vector<double>& b, std::vector<double> x);


// post-smoother operation
// input A an SPD matrix
// input b any vector in the image of A
// input x is a vector containing the return value y of the pre-smoother
// return value exactly z, where Az = b
template<typename T>
Vector<double> BackwardGaussS(SparseMatrix<T>& A, Vector<double>& b, std::vector<double> x);

//wrapper for forward and backward gauss in one function
Vector<double> FBGaussS(SparseMatrix<double>& A, Vector<double>& b);

//These Two-level and Multi-level PCG testers store P (or the sequence of P's
//in memory to save time, so these are for testing the speed of the preconditioner
template <typename T>
void PCGTL(SparseMatrix<T>& A, Vector<double>& x);

template <typename T>
void PCGML(SparseMatrix<T>& A, Vector<double>& x);

// This one is meant for use with the PCGML, where the sequence of P's stored in memory
// inverse action of a multi-level preconditioner
// input A the SPD laplacian
// input Lmax the maximum number of recursive calls this function can make
// input Ncoarse the upper bound on the smallest system to be solved directly 
// input nk the current coarseness of the system
// input k current iteration with k<=Lmax
// input q is the factor by which nk decreases at each iteration (currently unused)
// input r is the image in the system Mz = r
// return value is z
template <typename T>
Vector<double> MLNF2(SparseMatrix<T>& A, SparseMatrix<int>* P, int Lmax, int Ncoarse, int nk, int k, double q, Vector<double> r);


int SIZE = 0;

double kernel(int x, int y, int n)
{
	double temp = n;
	double step = (1.0/(temp-1));

	double yprime1, yprime2, xprime1, xprime2;
	double counter = 0.0;

	yprime1 = ((x-1)%n);

	for(int i=0; i<n; ++i)
	{
		if((x-((n)*i))>0)
		{
			counter = i;
		}
	}

	xprime1 = counter;

	yprime2 = ((y-1)%n);
	
	for(int i=0; i<n; ++i)	
	{
		if((y-((n)*i))>0)
		{ 
			counter = i;
		}
	}

	xprime2 = counter;
	
	yprime1 = yprime1*step;
	yprime2 = yprime2*step;
	xprime1 = xprime1*step;
	xprime2 = xprime2*step;

	double alpha = sqrt(((xprime1-xprime2)*(xprime1-xprime2))+((yprime1-yprime2)*(yprime1-yprime2)));
	
	double to_return = exp((-1.0)*alpha);
	return to_return; 
}

int main(int argc, char** argv)
{

	SparseMatrix<double> A;
	SparseMatrix<double> L;

	EdgeRead(L, A);
	
	Timer t1, t2, t3, t4, t5;

	Vector<double> x(L.Cols());
	x.Randomize();

	
	SparseMatrix<double> Jacobian = L.GetJacobi();
	SparseMatrix<double> GaussS = L.GetGaussS();
	
	t1.Click();
	CGtest(L, x);
	t1.Click();

	t2.Click();
	PCG_(L, x, Jacobian, JacobianInv);
	t2.Click();

	t3.Click();
	PCG_(L, x, GaussS, GaussSInv);
	t3.Click();
	
	t4.Click();
	//PCG_(L, x, L, TLInv);
	PCGTL(L, x);
	t4.Click();

	t5.Click();
	//PCG_(L, x, L, MLInv);
	PCGML(L, x);
	t5.Click();

    return EXIT_SUCCESS;
}

//base 0
double MaxRow(SparseMatrix<double>& Weights, int row)
{
	//note that this must be changed if random values
	//are not from -10.0 to 10.0
	double max = -20.0;

	for(int i=Weights.GetIndptr()[row]; i<Weights.GetIndptr()[row+1]; ++i)
	{
		max = std::max(Weights.GetData()[i], max);
	}

	return max;

}

//Weights square
SparseMatrix<int> LubysInterpolator(SparseMatrix<double>& Weights)
{
//SEQUENTIAL
	/*
	std::vector<double> newdata = Weights.GetData();
	std::vector<int> newindices = Weights.GetIndices();
	std::vector<int> newindptr = Weights.GetIndptr();
	
	std::vector<bool> Contract(newdata.size());
	
	double currentmax = 0;

	for(int i=0; i<Weights.Rows(); ++i)
	{
		currentmax = MaxRow(Weights, i);
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(newdata[j]>=std::max(currentmax, MaxRow(Weights, newindices[j])))
			{
				Contract[j] = true;
			}	
		}
	}

	int sum = 0;
	for(int i=0; i<Contract.size(); ++i)
	{
		if(Contract[i])
			++sum;
	}

	int numContract = sum/2;
	int offset = 0;

	CooMatrix<int> Interpolator(Weights.Rows(), Weights.Cols()-numContract);
	bool rowtrue = false;

	for(int i=0; i<Weights.Rows(); ++i)
	{
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(Contract[j])
			{
				
			rowtrue = true;

			if(i>newindices[j])
			{
				++offset;
			}
			else
			{
				Interpolator.Add(i, i-offset, 1);
				Interpolator.Add(newindices[j], i-offset, 1);
			}
			}

		}

		if(!rowtrue)
			Interpolator.Add(i, i-offset, 1);

		rowtrue = false;


	}

	return Interpolator.ToSparse();
*/

//PARALLEL
	std::vector<double> newdata = Weights.GetData();
	std::vector<int> newindices = Weights.GetIndices();
	std::vector<int> newindptr = Weights.GetIndptr();
	
	std::vector<bool> Contract(newdata.size());
	
	double currentmax = 0;

#pragma omp parallel for 
	for(int i=0; i<Weights.Rows(); ++i)
	{
		currentmax = MaxRow(Weights, i);
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(newdata[j]>=std::max(currentmax, MaxRow(Weights, newindices[j])))
			{
				Contract[j] = true;
			}	
		}
	}

	int sum = 0;
	for(int i=0; i<Contract.size(); ++i)
	{
		if(Contract[i])
			++sum;
	}

	int numContract = sum/2;
	int offset = 0;

	CooMatrix<int> Interpolator(Weights.Rows(), Weights.Cols()-numContract);
	bool rowtrue = false;

#pragma omp parallel for
	for(int i=0; i<Weights.Rows(); ++i)
	{
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(Contract[j])
			{
				
			rowtrue = true;

			if(i>newindices[j])
			{
				++offset;
			}
			else
			{
				Interpolator.Add(i, i-offset, 1);
				Interpolator.Add(newindices[j], i-offset, 1);
			}
			}

		}

		if(!rowtrue)
			Interpolator.Add(i, i-offset, 1);

		rowtrue = false;


	}

	return Interpolator.ToSparse();

}

//parallel version
SparseMatrix<int> PLubysInterpolator(SparseMatrix<double>& Weights)
{
	
	std::vector<double> newdata = Weights.GetData();
	std::vector<int> newindices = Weights.GetIndices();
	std::vector<int> newindptr = Weights.GetIndptr();
	
	std::vector<bool> Contract(newdata.size());
	
	double currentmax = 0;

#pragma openmp parallel for
	for(int i=0; i<Weights.Rows(); ++i)
	{
		currentmax = MaxRow(Weights, i);
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(newdata[j]>=std::max(currentmax, MaxRow(Weights, newindices[j])))
			{
				Contract[j] = true;
			}	
		}
	}

	int sum = 0;
	for(int i=0; i<Contract.size(); ++i)
	{
		if(Contract[i])
			++sum;
	}

	int numContract = sum/2;
	int offset = 0;

	CooMatrix<int> Interpolator(Weights.Rows(), Weights.Cols()-numContract);
	bool rowtrue = false;

#pragma openmp parallel for
	for(int i=0; i<Weights.Rows(); ++i)
	{
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(Contract[j])
			{
			
			rowtrue = true;
			if(i>newindices[j])
			{
				
				++offset;
			}
			else
			{
				Interpolator.Add(i, i-offset, 1);
				Interpolator.Add(newindices[j], i-offset, 1);
			}
			}
			
		
		}

		if(!rowtrue)

		rowtrue = false;


	}

	return Interpolator.ToSparse();

}

template <typename T>
Vector<double> SVM(SparseMatrix<T>& A, Vector<double>& x)
{
	
	
	std::vector<T> newdata = A.GetData();
	std::vector<int> newindptr = A.GetIndptr();
	std::vector<int> newindices = A.GetIndices();

	std::vector<double> newx = x.data();
	

//working without storing the vectors explicitly 
int N = A.Rows();
std::vector<double> X(N);
#pragma openmp parallel for num_threads(3) \ shared(N, X, A, x, newdata, newindptr, newindices) private(i, j, k, k1)
	for(int i=0; i<N; ++i)
	{
	
		double sum = 0;
		
		int k = A.GetIndptr()[i];
		int k1 = A.GetIndptr()[i+1];

		for(int j=k; j<k1; ++j)
		{
			sum += newx[newindices[j]]*newdata[j];
		}

		X[i] = sum;
	}
	

	Vector<double> toreturn2(X);
	return toreturn2;

}


//this function assumes a square matrix
template <typename T>
SparseMatrix<double> ProduceWeights(SparseMatrix<T>& A)
{

	CooMatrix<double> Temp(A.Cols());

	std::vector<double> newdata((A.GetData()).size());
	std::vector<int> newindices(A.GetIndices());
	std::vector<int> newindptr(A.GetIndptr());

 	std::random_device rd;
        std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10.0, 10.0);

	for(int i=0; i<A.Rows(); ++i)
	{
		
		for(int j=newindptr[i]; j<newindptr[i+1]; ++j)
		{
			if(newindices[j]==i)
			{
				Temp.AddSym(i, newindices[j], -100.0);
			}
			else
			{
			Temp.AddSym(i, newindices[j], dis(rd));
			}
		}

	}

	return Temp.ToSparse();

}


template <typename T>
Vector<double> MLNF(SparseMatrix<T>& A, int Lmax, int Ncoarse, int nk, int k, double q, Vector<double> r)
{


		if(nk<=Ncoarse || k == Lmax)
		{
			return CG(A, r, A.Cols(), 1e-6, 1e-6, false);
		}
		
		//x = 0
		Vector<double> x(A.Cols(), 0.0);
		
		SparseMatrix<double> Weights = ProduceWeights(A);
		SparseMatrix<int> P = LubysInterpolator(Weights);
		
		//pre-smooth
		std::vector<double> testx(A.Cols(), 0.0);
		x = ForwardGaussS(A, r, testx);
		Vector<double> rk = (P.Transpose()).Mult((r - A.Mult(x)));
		//Ak+1
		SparseMatrix<T> A2  = (P.Transpose()).Mult(A.Mult(P));

		//interpolation correction
		nk = P.Cols();
		x = x + P.Mult(MLNF(A2, Lmax, Ncoarse, nk, ++k, q, rk));
	
		//post-smooth
		Vector<double> temp2 = (r - A.Mult(x));
		x = x + BackwardGaussS(A, temp2, x.data());
		return x;
		
}


//these functions assume a square matrix
template<typename T>
Vector<double> ForwardGaussS(SparseMatrix<T>& A, Vector<double>& b, std::vector<double> x)
{

	std::vector<double> b2 = b.data();
	std::vector<double> diag = A.GetDiag();
	
	std::vector<int> tempIndptr = A.GetIndptr();
	std::vector<int> tempIndices = A.GetIndices();
	std::vector<T> tempData = A.GetData();
	
	
	for(int i=0; i<A.Cols(); ++i)
	{
		double sum = 0.0;


		for(int j=tempIndptr[i]; j<tempIndptr[i+1]; ++j)
		{
			if(tempIndices[j]!=i)
				sum -= (tempData[j]*x[tempIndices[j]]);
		}
		
		sum += b2[i];
		sum /= diag[i];
		x[i] = sum;


	}
	

	Vector<double> to_return(x);
	return to_return;

}



template<typename T>
Vector<double> BackwardGaussS(SparseMatrix<T>& A, Vector<double>& b, std::vector<double> x)
{
	
	std::vector<double> b2 = b.data();
	std::vector<double> diag = A.GetDiag();
	
	std::vector<int> tempIndptr = A.GetIndptr();
	std::vector<int> tempIndices = A.GetIndices();
	std::vector<T> tempData = A.GetData();

	
	for(int i=A.Cols()-1; i>=0; --i)
	{
		double sum = 0.0;
	

		for(int j=tempIndptr[i+1]-1; j>=tempIndptr[i]; --j)
		{
			if(tempIndices[j]!=i)
				sum -= (tempData[j]*x[tempIndices[j]]);
		}

		sum += b2[i];
		sum /= diag[i];
		x[i] = sum;

	}

	Vector<double> to_return(x);
	return to_return;
}

Vector<double> FBGaussS(SparseMatrix<double>& M, Vector<double>& b)
{
	std::vector<double> x(M.Cols(), 0.0);


	Vector<double> temp = ForwardGaussS(M, b, x);

	std::vector<double> temp2 = temp.data();

	Vector<double> toreturn = BackwardGaussS(M, b, temp2);


	return toreturn;
}

//Two-Level inverse action (non-functional)
template<typename T>
Vector<double> TLNF(SparseMatrix<T>& A, SparseMatrix<int>& P, SparseMatrix<T>& M, Vector<double> r)
{
	SparseMatrix<int> PT = P.Transpose();

	std::vector<double> X(r.size(), 0.0);
	//Action M^-1
	Vector<double> xOneThird = ForwardGaussS(A, r, X);
	Vector<double> rc = PT.Mult(r - A.Mult(xOneThird));

	//Action Ac^-1
	SparseMatrix<T> Ac = PT.Mult(A.Mult(P));
	Vector<double> xc = CG(Ac, rc, 1e-6, 1e-6, false);
	Vector<double> xTwoThird = (xOneThird + P.Mult(xc));

	//Action MT^-1
	Vector<double> temp = (r - A.Mult(xTwoThird) + (M.Transpose()).Mult(xTwoThird));

	Vector<double> x = BackwardGaussS(A, temp, xOneThird.data());

	return x;
	
}


void ParCGtest(SparseMatrix<double>& A, Vector<double>& x)
{
	
	Vector<double> b = SVM(A, x);
	int n = A.Cols();
	//initialize vectors and scalars to begin algorithm
	Vector<double> x0(n, 0.0);
	Vector<double> g(n, 0.0);
	Vector<double> r(b);
	Vector<double> p(r);

	double epsilon = 1.0e-6;
	double epsilon2 = 1.0e-12;
	double delta0 = b.Mult(b);
	double delta = delta0;
	double deltaOld, alpha, beta;
	
	//keep looping until this is done
	bool flag = true;

	//iterations
	int iter = 0;
	int iterMax = 1000;
	
	//the algorithm
	do
	{

		deltaOld = delta;

		g = SVM(A, p);

		double tau = p.Mult(g);
		alpha = (delta/tau);
		//x0 = ParAdd(x0, p, alpha);
		x.Add(alpha, p);
		//r = ParAdd(r, g, -alpha);
		r.Sub(alpha, g);
		//(6)
		delta = r.Mult(r);

		//(7)
		beta = (delta/deltaOld);
		
		//(8)
		p.Add((beta-1.0), p);
		p.Add(1, r);

		//(9)
		++iter;
		
		//can use this to test if each delta forms monotonic sequence
		//std::cout << delta << std::endl;

		if(iter>iterMax)
		{
			std::cout << "Convergence failure at: " << iter << " iterations.\n";
			return; //EXIT_FAILURE;
		}

		if(delta<((epsilon*epsilon)*delta0))
		{
			std::cout << "Success at: " << iter << " iterations.\n";
			//x0.Print("New x:");
			//b = A2.Mult(x0);
			//b.Print("New image:");
			
			return; //EXIT_SUCCESS;
		}
		
	}
	while(flag);
	

	

    return; //EXIT_SUCCESS;
}


void CGtest(SparseMatrix<double>& A, Vector<double>& x)
{
	Vector<double> b = A.Mult(x);
	int n = A.Cols();
	//initialize vectors and scalars to begin algorithm
	Vector<double> x0(n, 0.0);
	Vector<double> g(n, 0.0);
	Vector<double> r(b);
	Vector<double> p(r);

	double epsilon = 1.0e-6;
	double epsilon2 = 1.0e-12;
	double delta0 = b.Mult(b);
	double delta = delta0;
	double deltaOld, alpha, beta;
	
	//keep looping until this is done
	bool flag = true;

	//iterations
	int iter = 0;
	int iterMax = 1000;
	
	//the algorithm
	do
	{
		//(1)
		deltaOld = delta;

		//(2)
		g = A.Mult(p);

		//(3)
		double tau = p.Mult(g);
		alpha = (delta/tau);

		//(4)
		x0.Add(alpha, p);

		//(5)
		r.Sub(alpha, g);

		//(6)
		delta = r.Mult(r);

		//(7)
		beta = (delta/deltaOld);
		
		//(8)
		p.Add((beta-1.0), p);
		p.Add(1, r);

		//(9)
		++iter;
		
		//can use this to test if each delta forms monotonic sequence
		//std::cout << delta << std::endl;

		if(iter>iterMax)
		{
			std::cout << "Convergence failure at: " << iter << " iterations.\n";
			return; //EXIT_FAILURE;
		}

		if(delta<((epsilon*epsilon)*delta0))
		{
			std::cout << "Success at: " << iter << " iterations.\n";
			//x0.Print("New x:");
			//b = A2.Mult(x0);
			//b.Print("New image:");
			
			return; //EXIT_SUCCESS;
		}
		
	}
	while(flag);
	

	

    return; //EXIT_SUCCESS;
}

Vector<double> JacobiInverse(SparseMatrix<double>& A, Vector<double>& x)
{
	Vector<double> toreturn(A.Cols());
	std::vector<double> data = A.GetData();

	for(int i=0; i<x.size(); ++i)
	{
		toreturn[i] = x[i]*(1/data[i]);
	}

	return toreturn;
}


template <typename T>
void PCG_(SparseMatrix<T>& A, Vector<double>& x, SparseMatrix<T>& M, Vector<double> (*precond)(SparseMatrix<T>& M, Vector<double>& r))
{

	Vector<double> b = A.Mult(x); 

	
	Vector<double> x0(SIZE, 0.0);
	Vector<double> g(SIZE, 0.0);
	Vector<double> r(b);
	Vector<T> rTilda(SIZE, 0.0);
	Vector<int> partition(SIZE, 0);

	double epsilon, tau, alpha, beta, delta0, deltaOld, delta;
	int iter, iterMax;

	epsilon = 1e-6;
	iter = 0;
	iterMax = SIZE;
	
	//set initial rTilda
	rTilda = precond(M, r);
	delta0 = rTilda.Mult(r);
	delta = delta0;
	Vector<double> p(rTilda);
	
	bool flag = true; //for while loop

	do
	{
		//(1)
		//g = A.Mult(p);
		g = SVM(A, p);

		//(2)
		tau = p.Mult(g);

		//(3)
		alpha = delta/tau;

		//(4)
		x0 = x0 + alpha*p;

		//(5)
		r = r - alpha*g;

		//(6)
		rTilda = precond(M, r);


		//(7)
		deltaOld = delta;

		//(8)
		delta = rTilda.Mult(r);

		//(9)
		++iter;

		if(delta<(epsilon*epsilon*delta0))
		{
			std::cout << "Success at " << iter << " iterations.\n";
			flag = false;
		}
		if(iter>iterMax)
		{
			std::cout << "Convergence failure.\n";
			flag = false;
		}

		//(10)
		beta = delta/deltaOld;

		//(11)
		p = rTilda + beta*p;



	}
	while(flag);
}

//row parameter base 0
template<typename T> 
int RowSum(SparseMatrix<T>& A, int row)
{
	int sum = 0;
	std::vector<T> newdata = A.GetData();
	std::vector<int> newindices = A.GetIndices();
	std::vector<int> newindptr = A.GetIndptr();

	for(int i=newindptr[row]; i<newindptr[row+1]; ++i)
	{
		if(newdata[i]!=0 && newindices[i]!=i)
			++sum;
	}

	return sum;
}

//assumes diagonal is always present
template<typename T>
SparseMatrix<T> ModLaplacian(SparseMatrix<T>& A)
{
	std::vector<T> data = A.GetData();
	std::vector<int> indptr = A.GetIndptr();
	std::vector<int> indices = A.GetIndices();
	
	std::vector<T> newdata(data.size());
	
	for(int i=0; i<A.Rows(); ++i)
	{
		
		for(int j=indptr[i]; j<indptr[i+1]; ++j)
		{
			if(indices[j]==i)
			{
				newdata[j] = RowSum(A, i);

			}
			else
			{
				newdata[j] = -1;
			}
		}
	
	}

	SparseMatrix<T> ToReturn(indptr, indices, newdata, A.Rows(), A.Cols());
	ToReturn.EliminateRowColD(A.Rows()-1);
	return ToReturn;

}


template <typename T>
void EdgeRead(SparseMatrix<T>& Laplacian, SparseMatrix<T>& Adjacency/* bool MTX*/)
{
	//choose which edgelist to use
	//std::fstream inFile("as20000102.edges");
	std::fstream inFile("amazon_h_1.edgelist");

	if(!inFile)
	{
		std::cout << "Error: edge list not found.\n";
	}

	int n; //size of V(G)
	int loop; //how long we will read from file (how many lines)
	
	inFile >> n;
	if(/*MTX*/1)
	{

	}

	SIZE = n;
	inFile >> loop;

	int data1, data2; //vertices v1 and v2, indicating who are adjacent 
	CooMatrix<T> A(n);
	CooMatrix<T> A3(n);

	for(int i=0; i<loop; ++i)
	{
		inFile >> data1;
		inFile >> data2;

		A.Add(data1, data2, -1);
		A3.Add(data1, data2, 1);
		
		if(i<n)
		{
			A.Add(i, i, -1); //add the diagonal (loops)
			A3.Add(i, i, 1);
		}
	}
	
	inFile.close();
	
	Adjacency = A3.ToSparse();
	
	//construct Laplacian, which is L = D - A
	DenseMatrix A2 = A.ToDense(); //the dense matrix has a "GetRow" function
	Vector<T> One(n, 1.0);
	VectorView<T> OneV(One);
	VectorView<T> Row(OneV);
	
	for(int i=0; i<n; ++i)
	{
		Row.Set(0.0, OneV);
		A2.GetRow(i, Row);
		int DegreeSum = Row.Mult(One);
		A.Add(i, i, DegreeSum);
	}
	

	SparseMatrix<T> L = A.ToSparse();
	L.EliminateRowColD(n-1); //delete off-diagnoal entries in last row/col

	Laplacian = L;
}

template <typename T>
Vector<double> JacobianInv(SparseMatrix<T>& M, Vector<double>& r)
{
	return JacobiInverse(M, r);
}


template <typename T>
Vector<double> GaussSInv(SparseMatrix<T>& M, Vector<double>& r)
{
	return FBGaussS(M, r);
}


//will pass A as M because M is generated each time
template <typename T>
Vector<double> TLInv(SparseMatrix<T>& M, Vector<double>& r)
{
	int size = std::cbrt(r.size());
	SparseMatrix<double> Weights = ProduceWeights(M);
	SparseMatrix<int> P = LubysInterpolator(Weights);

	return TLNF(M, P, M, r);
}


template <typename T>
Vector<double> MLInv(SparseMatrix<T>& M, Vector<double>& r)
{
	
	int Ncoarse = std::cbrt(r.size());
	return MLNF(M, 40, Ncoarse, r.size(), 0, 0.5, r);
}


template <typename T>
void PCGTL(SparseMatrix<T>& A, Vector<double>& x)
{

	Vector<double> b = A.Mult(x);
	
	SparseMatrix<double> Weights = ProduceWeights(A);
	SparseMatrix<int> P = LubysInterpolator(Weights);
	
	Vector<double> x0(SIZE, 0.0);
	Vector<double> g(SIZE, 0.0);
	Vector<double> r(b);
	Vector<T> rTilda(SIZE, 0.0);
	Vector<int> partition(SIZE, 0);

	double epsilon, tau, alpha, beta, delta0, deltaOld, delta;
	int iter, iterMax;

	epsilon = 1e-6;
	iter = 0;
	iterMax = SIZE;
	
	//set initial rTilda
	rTilda = TLNF(A, P, A, r);
	delta0 = rTilda.Mult(r);
	delta = delta0;
	Vector<double> p(rTilda);
	
	bool flag = true; //for while loop

	do
	{
		//(1)
		//g = A.Mult(p);
		g = SVM(A, p);

		//(2)
		tau = p.Mult(g);

		//(3)
		alpha = delta/tau;

		//(4)
		x0 = x0 + alpha*p;

		//(5)
		r = r - alpha*g;

		//(6)
		rTilda = TLNF(A, P, A, r);


		//(7)
		deltaOld = delta;

		//(8)
		delta = rTilda.Mult(r);

		//(9)
		++iter;

		if(delta<(epsilon*epsilon*delta0))
		{
			std::cout << "Success at " << iter << " iterations.\n";
			flag = false;
		}
		if(iter>iterMax)
		{
			std::cout << "Convergence failure.\n";
			flag = false;
		}

		//(10)
		beta = delta/deltaOld;

		//(11)
		p = rTilda + beta*p;



	}
	while(flag);
}

template <typename T>
void PCGML(SparseMatrix<T>& A, Vector<double>& x)
{

	Vector<double> b = A.Mult(x); 
	
	SparseMatrix<int>* Pseq = new SparseMatrix<int>[40];
	SparseMatrix<double> Weights;
	SparseMatrix<T> Ac = A;

	for(int i=0; i<40; ++i)
	{
		Weights = ProduceWeights(Ac);
		Pseq[i] = LubysInterpolator(Weights);
		Ac = (Pseq[i].Transpose()).Mult(Ac.Mult(Pseq[i]));
	}

	Vector<double> x0(SIZE, 0.0);
	Vector<double> g(SIZE, 0.0);
	Vector<double> r(b);
	Vector<T> rTilda(SIZE, 0.0);
	Vector<int> partition(SIZE, 0);

	double epsilon, tau, alpha, beta, delta0, deltaOld, delta;
	int iter, iterMax;

	epsilon = 1e-6;
	iter = 0;
	iterMax = SIZE;
	int cube = std::cbrt(r.size());

	//set initial rTilda
	//rTilda = precond(M, r);
	rTilda = MLNF2(A, Pseq, 40, cube, r.size(), 0, 0.5, r);
	delta0 = rTilda.Mult(r);
	delta = delta0;
	Vector<double> p(rTilda);
	
	bool flag = true; //for while loop

	do
	{
		//(1)
		//g = A.Mult(p);
		g = SVM(A, p);

		//(2)
		tau = p.Mult(g);

		//(3)
		alpha = delta/tau;

		//(4)
		x0 = x0 + alpha*p;

		//(5)
		r = r - alpha*g;

		//(6)
		//rTilda = precond(M, r);
		rTilda = MLNF2(A, Pseq, 40, cube, r.size(), 0, 0.5, r);

		//(7)
		deltaOld = delta;

		//(8)
		delta = rTilda.Mult(r);

		//(9)
		++iter;

		if(delta<(epsilon*epsilon*delta0))
		{
			std::cout << "Success at " << iter << " iterations.\n";
			flag = false;
		}
		if(iter>iterMax)
		{
			std::cout << "Convergence failure.\n";
			flag = false;
		}

		//(10)
		beta = delta/deltaOld;

		//(11)
		p = rTilda + beta*p;



	}
	while(flag);
}


template <typename T>
Vector<double> MLNF2(SparseMatrix<T>& A, SparseMatrix<int>* P, int Lmax, int Ncoarse, int nk, int k, double q, Vector<double> r)
{


		if(nk<=Ncoarse || k == Lmax)
		{
			return CG(A, r, A.Cols(), 1e-6, 1e-6, false);
		}
		
		//x = 0
		Vector<double> x(A.Cols(), 0.0);
		
		//pre-smooth
		std::vector<double> testx(A.Cols(), 0.0);
		x = ForwardGaussS(A, r, testx);
		Vector<double> rk = (P[k].Transpose()).Mult((r - A.Mult(x)));
		//Ak+1
		SparseMatrix<T> A2  = (P[k].Transpose()).Mult(A.Mult(P[k]));

		//interpolation correction
		nk = P[k].Cols();
		x = x + P[k].Mult(MLNF2(A2, P, Lmax, Ncoarse, nk, ++k, q, rk));
	
		//post-smooth
		Vector<double> temp2 = (r - A.Mult(x));
		x = x + BackwardGaussS(A, temp2, x.data());
		return x;
}

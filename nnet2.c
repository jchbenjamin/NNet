#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EEE 2.718281828

#define NUMINPUTS 2
#define SAMPLESIZE 314
#define LAYERONESIZE 10
#define LAYERTWOSIZE 1

typedef struct node {
	double b;
	double v;
	double y;
	double e;
	double energy;
	double del;
} node;

double phi(double v)
{
	return (1 / (1 + pow( EEE , -v)));
}

double dphi(double v)
{
	double p = phi(v);
	return ( p * ( 1.0 - p) );
}

void calc_error(node* k, int d) 
{
	k->e = ( ((double) d) - k->y );
}

void calc_energy(node* k)
{
	k->energy = 0.5 * pow(k->e, 2);
}

void calc_y(node* k) // get y from v based on sigmoid activation function
{
	k->y = phi(k->v);
}

void calc_del_out(node* k)
{
	k->del = k->e * dphi(k->v);
}

void calc_del_hid(node* k, int nodenum, double w[LAYERONESIZE][LAYERTWOSIZE], node layer[LAYERTWOSIZE])
{
	double d = 0.0;
	double p = dphi(k->v);
	double q = 0.0;
	int i=0;
	for(i=0;i<LAYERTWOSIZE;i++)
	{
		q += (layer[i].del * w[nodenum][i]);
	}
	k->del = p*q;
}

void calc_vy_from_input(node* n, int nodenum, double w[NUMINPUTS][LAYERONESIZE], double x[SAMPLESIZE][NUMINPUTS], int xsamp) {
	double sum = n->b; //initialize sum to b;
//			printf("fromx:: b: %f  ", sum);
	int i = 0;
	for(i=0; i< NUMINPUTS; i++)
	{
			//printf("w1[%d][%d]: %f x[%d][%d]: %f",i,nodenum,w[i][nodenum],xsamp,i,x[xsamp][i]);
			sum += ( w[i][nodenum] * x[xsamp][i] ); ///OKAY HERE	
	}
	n->v = sum;
//			printf("v: %f  ", n->v);
	calc_y(n);
//			printf("y: %f\n", n->y);
}

void calc_vye_from_layer(node* n, int nodenum, double w[LAYERONESIZE][LAYERTWOSIZE], node layer[LAYERONESIZE], int d[SAMPLESIZE], int xsamp) {
	double sum = n->b;
//	printf("hidden:: b: %f  ", sum);
	int i = 0;
	for(i=0; i< LAYERONESIZE ; i++)
	{
			sum += ( w[i][nodenum] * ( layer[i].y ) );
	}
	n->v = sum;
//	printf("v: %f  ", n->v);
	calc_y(n);
//			printf("y: %f  ", n->y); 
	calc_error(n,d[xsamp]);
	calc_energy(n);
//			printf("e: %f  ", n->e);
	calc_del_out(n);
//			printf("del: %f\n", n->del);
		
}

void feedforward1(double w[NUMINPUTS][LAYERONESIZE], node layer[LAYERONESIZE], double x[SAMPLESIZE][NUMINPUTS], int xsamp)
{
	int i = 0;

	for(i; i < LAYERONESIZE; i++)
	{
		calc_vy_from_input(&(layer[i]), i, w, x, xsamp);
	}

}

void feedforward2(double w[LAYERONESIZE][LAYERTWOSIZE], node outlayer[LAYERTWOSIZE], node inlayer[LAYERONESIZE], int d[SAMPLESIZE], int xsamp)
{
	int i = 0;
	for(i; i < LAYERTWOSIZE; i++)
	{
		calc_vye_from_layer(&(outlayer[i]), i, w, inlayer, d, xsamp);
	}		
}

void backprop(double w[LAYERONESIZE][LAYERTWOSIZE], node layer1[LAYERONESIZE], node layer2[LAYERTWOSIZE])
{
	int i = 0;
	for(i; i<LAYERONESIZE;i++)
	{
		calc_del_hid(&(layer1[i]),i,w,layer2);
	}
}

void updateweights_all(double alpha, double beta, double w1[NUMINPUTS][LAYERONESIZE], double w2[LAYERONESIZE][LAYERTWOSIZE], double old_dw[LAYERONESIZE][LAYERTWOSIZE], node layer1[LAYERONESIZE], node layer2[LAYERTWOSIZE], double x[SAMPLESIZE][NUMINPUTS], int xsamp)
{
	int i=0;
	int j=0;
	//BIASES
	for(i=0;i<LAYERONESIZE;i++)
	{
		layer1[i].b += alpha * layer1[i].del;
	}
	for(i=0;i<LAYERTWOSIZE;i++)
	{
		layer2[i].b += alpha * layer2[i].del;
	}
	//weights
	for(i=0;i<NUMINPUTS;i++)
	{
		for(j=0;j<LAYERONESIZE;j++)
		{
			w1[i][j] += alpha * layer1[j].del * x[xsamp][i];
		}
	}
	
	for(i=0;i<LAYERONESIZE;i++)
	{
		for(j=0;j<LAYERTWOSIZE;j++)
		{

			old_dw[i][j] = beta * old_dw[i][j] + alpha * layer2[j].del * layer1[i].y;
			w2[i][j] += old_dw[i][j];
		}
	}		

}

void avg_err_energy(double* avg_error, node layer[LAYERTWOSIZE], double err[SAMPLESIZE], int sampi)
{
	double e = 0.0;
	double sum = 0.0;	
	int i = 0;
	for(i=0; i<LAYERTWOSIZE; i++)
	{
		e += layer[i].energy;
	}
	err[sampi] = e;
	//printf("error at %d: %f\n",sampi,err[sampi]);
	if(sampi == (SAMPLESIZE-1))
	{ //WE ARE READY TO GET AVERAGE SINCE EPOCH DONE
		e = 0.0;
		for(i=0;i<SAMPLESIZE;i++)
		{
			e += err[i];
			//printf("sum at %d: %f\n",i,e);
		}

		*avg_error = ( (double) 1/SAMPLESIZE) * e;
	}
}

void fill_w1(FILE *f, double w[NUMINPUTS][LAYERONESIZE])
{
	char line[512];
	int i;
	int j;
	for(j = 0; j < LAYERONESIZE; j++)
	{
		i=0;
		fgets(line, 512, f);

		w[0][j] = (double) atof(strtok(line," ,\n"));	
		//printf ("w%d%d:%f",i,j,w[0][j]);
		for(i=1; i<(NUMINPUTS); i++)
		{
			w[i][j] = (double) atof(strtok(NULL," ,\n"));
		//		printf ("w%d%d:%f",i,j,w[i][j]);
		}
		
	}
	//printf("\n");
fclose(f);
}

void fill_w2(FILE *f, double w[LAYERONESIZE][LAYERTWOSIZE])
{

	char line[512];
	int i;
	int j;
	for(j = 0; j < LAYERTWOSIZE; j++)
	{
		i=0;
		fgets(line, 512, f);

		w[i][0] = (double) atof( (char*) strtok(line," ,\n"));	
	//					printf ("w%d%d:%f ",i,j,w[i][j]);
	for(i=1; i<(LAYERONESIZE); i++)
		{
			w[i][j] = (double) atof( (char*) strtok(NULL," ,\n"));
	//						printf ("w%d%d:%f",i,j,w[i][j]);
		}
	
	}
	//printf("\n");
	fclose(f);

}

void fill_b1(FILE *f, node l[LAYERONESIZE])
{
	char line[512];
	int i;
	int j;
	for(i = 0; i < LAYERONESIZE; i++)
	{
		fgets(line, 512, f);
		l[i].b = atof(strtok(line," ,\n"));	
	//	printf ("b1%d:%f ",i,l[i].b);
	}
	//printf("\n");
	fclose(f);
}

void fill_b2(FILE *f, node l[LAYERTWOSIZE])
{
	char line[512];
	int i;
	int j;
	for(i = 0; i < LAYERTWOSIZE; i++)
	{
		fgets(line, 512, f);
		l[i].b = (double) atof(strtok(line," ,\n"));	
	//			printf ("b2%d:%f ",i,l[i].b);
	}
	//printf("\n");
	fclose(f);
}

void fill_x(FILE *f, double x[SAMPLESIZE][NUMINPUTS], int d[SAMPLESIZE])
{
	char line[512];
	int i=0;
	int j=0;
	for(i = 0; i < SAMPLESIZE; i++)
	{
		fgets(line, 512, f);

		x[i][0] = atof(strtok(line, " ,\n"));	
		//printf("x[%d][0]: %f\n",i,x[i][0]);
		for(j=1; j<(NUMINPUTS); j++)
		{
			x[i][j] = (double) atof(strtok(NULL," ,\n"));
		
		//	printf("WOOPx[%d][%d]: %f\n",i,j,x[i][j]);
		}
		
		d[i] = (int) atoi(strtok(NULL," ,\n"));
		printf("d%d: %d\n",i,d[i]);
	}
fclose(f);
}

void printw(double w1[NUMINPUTS][LAYERONESIZE], double w2[LAYERONESIZE][LAYERTWOSIZE])
{
	int i=0;
	int j=0;
	printf("w1: ");
	for(i=0;i<NUMINPUTS;i++){
		for(j=0;j<LAYERONESIZE;j++){
			printf("%f ",w1[i][j]);
		}
	}
	printf("\n");
	
	printf("w2: ");
	for(i=0;i<LAYERONESIZE;i++){
		for(j=0;j<LAYERTWOSIZE;j++){
			printf("%f ",w2[i][j]);
		}
	}
	printf("\n");

}

int main(int argc, char** argv)
{
	double avg_err = 0.0;
	double alpha = 0.7; //learning rate
	double beta = 0.3; //momentum
	int a; int b;
	int i=0;
	int j=0;
	double x[SAMPLESIZE][NUMINPUTS];

	double err[SAMPLESIZE];
	
	int d[SAMPLESIZE];
	
	node layer1[LAYERONESIZE]; //layer1
	
	node layer2[LAYERTWOSIZE]; //layer2
	double w1[NUMINPUTS][LAYERONESIZE]; //weights for links from input to layer1
	for(a=0; a<NUMINPUTS; a++)
	{
		for(b=0; b<LAYERONESIZE; b++)
		{
		//	w1[a][b] = 1.0 + (0.5/(a+b+1.0));
		w1[a][b] = 0.0;
		}
	}
	
	double w2[LAYERONESIZE][LAYERTWOSIZE];
for(a=0; a<LAYERONESIZE; a++)
	{
		for(b=0; b<LAYERTWOSIZE; b++)
		{
			w2[a][b] = 0.0;
		}
	}

	double old_dw[LAYERONESIZE][LAYERTWOSIZE];
	for(a=0; a<LAYERONESIZE; a++)
	{
		for(b=0; b<LAYERTWOSIZE; b++)
		{
			old_dw[a][b] = 0.0;
		}
	}

//FILL VARS
	FILE *f = fopen("cross_data/cross_data.csv", "r");
	if (f == NULL) {
		printf("file not open\n");
		return 1;
	}
	//		printf("fillx\n");
	fill_x(f, x, d);
	
	f = fopen("cross_data/w1.csv", "r");
	if (f == NULL) {
		printf("file not open\n");
		return 1;
	}
	//			printf("fillw1\n");
	fill_w1(f, w1);
	f = fopen("cross_data/w2.csv", "r");
	if (f == NULL) {
		printf("file not open\n");
		return 1;
	}
	//				printf("fillw2\n");
	fill_w2(f, w2);
		
	f = fopen("cross_data/b1.csv", "r");
	if (f == NULL) {
		printf("file not open\n");
		return 1;
	}
	fill_b1(f, layer1);
	f = fopen("cross_data/b2.csv", "r");
	if (f == NULL) {
		printf("file not open\n");
		return 1;
	}
	fill_b2(f, layer2);
	
	
	


int maini=0;
int mainj=0;
//MAIN LOOP	GOES THROUGH ONE EPOCH

do { 
for(maini=0;maini<SAMPLESIZE;maini++){
//run one round
		feedforward1(w1, layer1, x, i);
		//printf("FEEDF: %f\n", (w1[0][0]));
		feedforward2(w2, layer2, layer1, d, i);
		//printf("FEEDF: %f\n", w2[0][0]);
		backprop(w2, layer1, layer2);
		//printf("backprop: %f %f\n", w1[0][0],w2[0][0]);
		//printf("layer1: %f %f %f %f layer2: %f %f %f %f\n", layer1[0].b, layer1[0].v, layer1[0].y, layer1[0].del, layer2[0].b,layer2[0].v,layer2[0].y,layer2[0].del);
		updateweights_all(alpha, beta, w1,w2,old_dw,layer1,layer2,x,i);
			avg_err_energy((double*)&avg_err,layer2,err,maini);

}
	printw(w1,w2);
	printf("b: %f y:%f\n",layer2[0].b,layer2[0].y);
	printf("error: %f\n", avg_err);
} while(avg_err >= 0.001);


for(i=0;i<SAMPLESIZE;i++)
{
//now classify a point
feedforward1(w1,layer1,x,i);
feedforward2(w2,layer2,layer1,d,i);
printf("%d e: %f y: %f d: %d\n",i,layer2[i].e,layer2[i].y,d[i]);
}

return 0;
}


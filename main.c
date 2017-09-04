   #include<stdio.h>
#include <time.h>


//int c_conv ( int in_channel, int o_channel, int kernel_size, int stride);

int  c_conv() {
    
    int c=3,h=1280, w=720;
	int C, H, W;
	int inputImg[c][h][w];
    time_t t;
    srand((unsigned)time(&t));
    for(C=0;C<c;C++){
	for(H=0;H<h;H++)
    {
        for(W=0;W<w;W++)
            {
                inputImg[C][H][W] = (rand()%255);
                printf("%d   ",inputImg[C][H][W]);
            }

        printf("\n");
    }
	}
     clock_t start, end;
    double cpu_time_used;
    start = clock();
	
	
      int in_channel=3;
      int o_channel=1;
      int kernel_size=3;
      int stride=1;  
      //int inputImg [3][3][3]={{{1,2,3},{4,5,6},{7,8,9}},{{1,2,3},{4,5,6},{7,8,9}},{{1,2,3},{4,5,6},{7,8,9}}} ;
      //int inputImg [3][5][5]={{{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{7,8,9,10,11},{7,8,9,10,11}},{{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{7,8,9,10,11},{7,8,9,10,11}},{{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{7,8,9,10,11},{7,8,9,10,11}}} ;
	  int k [3][3][3]= {{{-1,-1,-1},{0,0,0},{1,1,1}},{{-1,-1,-1},{0,0,0},{1,1,1}},{{-1,-1,-1},{0,0,0},{1,1,1}}};
	  int k2 [2][3][4]= {{{-1,-1,-1,4},{0,0,0,4},{1,1,1,4}},{{-1,-1,-1,4},{0,0,0,4},{1,1,1,4}}};
	  
	  int k4 [5][5];
      /*  printf("total elements=%d \n",sizeof(inputImg)/sizeof(int));
        printf("total elements in one channel=%d \n",sizeof(inputImg[0])/sizeof(int));
        printf("total elements in one row> #column =%d \n",sizeof(inputImg[0][0])/sizeof(int));
        printf("total elements in one column> #row =%d \n",sizeof(inputImg[0])/sizeof(inputImg[0][0]));*/
	  int input_x=sizeof(inputImg[0][0])/sizeof(int);
	  int input_y=sizeof(inputImg[0])/sizeof(inputImg[0][0]);
	  int p=(input_x-kernel_size)/stride+1;
	  int q=(input_y-kernel_size)/stride+1;
	  int r=in_channel;
	  int s=o_channel;
	  printf("r =%d, s =%d, p =%d, q =%d \n", r, s, p, q );
      int d[s][p][q];
      int e,f,g;
	    for (e=0;e<s;e++){
            for (f=0;f<p;f++){
                for (g=0;g<q;g++){
                    d[e][f][g]=0;
                    //printf("%d",d[e][f][g]);
                }
            }
        }
     int o,l,m,n,i,j;
     for ( o=0; o<s; o++){
		for ( l=0; l<r; l++){
			for ( m = 0; m < p; m++) {
				for ( n = 0; n < q; n++) { 
					for ( i = 0; i < kernel_size; i++) {
						for ( j = 0; j < kernel_size; j++) {
						    //printf("l=%d,  i+m+(stride-1)=%d, j+n+(stride-1)=%d \n", l,i+m+(stride-1),j+n+(stride-1));
						    //printf("o=%d,  kernel_size-i=%d, kernel_size-j=%d \n", o,kernel_size-i-1,kernel_size-j-1);
						    //printf("in=%d,  k=%d, in*k=%d \n", inputImg[l][i+m+(stride-1)][j+n+(stride-1)],k[o][kernel_size-i-1][kernel_size-j-1],(inputImg[l][i+m+(stride-1)][j+n+(stride-1)])*(k[o][kernel_size-i-1][kernel_size-j-1]) );
							
							d[o][m][n] += ((inputImg[l][i+m*stride][j+n*stride])*(k[o][kernel_size-i-1][kernel_size-j-1]));
							//printf("d=%d\n",d[o][m][n] );
						}
					}
				}
			}
		}
	}
	    for (e=0;e<s;e++){
            for (f=0;f<p;f++){
                for (g=0;g<q;g++){
                    //printf("%d  ",d[e][f][g]);
                }
                //printf("\n");
            }
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("%d \n",cpu_time_used );
      return ((kernel_size*kernel_size)*p*q*in_channel*o_channel); 
        
}





void main () {
    int a;
    a=c_conv();
    printf("result %d \n",a);  
}




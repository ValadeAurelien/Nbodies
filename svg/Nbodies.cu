#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <SDL/SDL.h>

#define pot_const 1e-6
#define BLOCKSIZE 16
#define OFFSET 10
#define AUXSIGMA 3

typedef struct coord_t {float x, y, m;} coord_t;

__device__
coord_t potential(coord_t contrib, coord_t a, coord_t b)
{
  contrib.x += pot_const * b.m / powf( sqrtf( powf((a.x-b.x), 2) + powf(a.y-b.y, 2) ), 3)*(b.x-a.x);
  contrib.y += pot_const * b.m / powf( sqrtf( powf((a.x-b.x), 2) + powf(a.y-b.y, 2) ), 3)*(b.y-a.y);
  return contrib;
}


__global__ 
void init_coordinates_n_masses(coord_t * bodies_1, coord_t * bodies_2, curandState_t * states, size_t X, 
                               float max_X, float max_Y, float min_m, float max_m, int seed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  curand_init(seed, j*X+i, OFFSET, &states[j*X+i]);
  bodies_1[j*X+i].x = curand_uniform(&states[j*X+i])*max_X;
  bodies_1[j*X+i].y = curand_uniform(&states[j*X+i])*max_Y;
  bodies_1[j*X+i].m = min_m+ curand_uniform(&states[j*X+i]) * (max_m-min_m);
  //bodies_1[j*X+i].m = min_m+ curand_log_normal(&states[j*X+i], (max_m+min_m)/2, (max_m-min_m)/AUXSIGMA);
  bodies_2[j*X+i].m = bodies_1[j*X+i].m;
}

__global__ 
void update_bodies(coord_t * bodies_in, coord_t * bodies_out, size_t X, size_t Y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  coord_t contrib = {0,0};
  for(int h=0; h<Y; h++)
    for(int w=0; w<X; w++)
      if ( h != j && w != i)
        contrib = potential(contrib, bodies_in[j*X+i], bodies_in[h*X+w]);
  bodies_out[j*X+i].x = bodies_in[j*X+i].x + contrib.x;
  bodies_out[j*X+i].y = bodies_in[j*X+i].y + contrib.y;

}

void update_pixels(coord_t * h_bodies, Uint32 * pixels, float max_X, float max_Y, float min_m, float max_m, 
                   size_t X, size_t Y, size_t nb_pts_X, size_t nb_pts_Y)
{
  size_t px, py;
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++)
    {
      px = ceil( nb_pts_X * h_bodies[y*X+x].x / max_X);
      py = ceil( nb_pts_Y * h_bodies[y*X+x].y / max_Y);
      if (0<py && py<nb_pts_Y && 0<px && px< nb_pts_X)
        pixels[py*nb_pts_X+px] = floor( (16711680-65280) * (h_bodies[y*X+x].m - min_m) / max_m + 65280);
    }
}

int main(int argc, char * argv[])
{
  if (argc != 8)
  {
    printf("nb bodies, nb_pts_X, nb_pts_Y, max_X, max_Y, min_m, max_m\n");
    return EXIT_SUCCESS;
  }

  size_t N = atoi(argv[1]),
         nb_pts_X  =  atoi(argv[2]),
         nb_pts_Y = atoi(argv[3]);
  float max_X = atof(argv[4]),
        max_Y = atof(argv[5]),
        min_m = atof(argv[6]),
        max_m = atof(argv[7]);

  size_t X, Y; 
  X = ceil(sqrt(N));
  Y = ceil((float) N/X);

  printf("X, Y = %d, %d\n", X, Y);
  
  dim3 blockSize (BLOCKSIZE,BLOCKSIZE);
  dim3 gridSize (ceil( (float) X/BLOCKSIZE), ceil( (float) Y/BLOCKSIZE));

  SDL_Init(SDL_INIT_VIDEO);
  SDL_Surface *SDL_img = SDL_SetVideoMode(nb_pts_X, nb_pts_Y, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
  SDL_Event event;

  coord_t *h_bodies = (coord_t*) malloc(X*Y*sizeof(coord_t));

  coord_t *d_bodies_1, *d_bodies_2;
  curandState *states;
  
  cudaMalloc(&d_bodies_1, X*Y*sizeof(coord_t));
  cudaMalloc(&d_bodies_2, X*Y*sizeof(coord_t));
  cudaMalloc(&states, X*Y*sizeof(curandState));

  init_coordinates_n_masses<<<gridSize, blockSize>>>(d_bodies_1, d_bodies_2, states, X,
                                                   max_X, max_Y, min_m, max_m, time(NULL));
  cudaMemcpy(h_bodies, d_bodies_1, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
  update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, max_X, max_Y, min_m, max_m, X, Y, nb_pts_X, nb_pts_Y);
  SDL_Flip(SDL_img);
  for (int i=0; i<X*Y; i++) printf("%f %f %f | ", h_bodies[i].x, h_bodies[i].y, h_bodies[i].m); printf("\n");

  while (true)
  {
    update_bodies<<<gridSize, blockSize>>>(d_bodies_1, d_bodies_2, X, Y);
    cudaMemcpy(h_bodies, d_bodies_2, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
    update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, max_X, max_Y, min_m, max_m, X, Y, nb_pts_X, nb_pts_Y);
    SDL_Flip(SDL_img);

    update_bodies<<<gridSize, blockSize>>>(d_bodies_2, d_bodies_1, X, Y);
    cudaMemcpy(h_bodies, d_bodies_1, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
    //for (int i=0; i<X*Y; i++) printf("%f %f %f | ", h_bodies[i].x, h_bodies[i].y, h_bodies[i].m); printf("\n");
    update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, max_X, max_Y, min_m, max_m, X, Y, nb_pts_X, nb_pts_Y);
    SDL_Flip(SDL_img);
    if ( SDL_PollEvent(&event) )
      switch (event.type)
      {
        default :
          break;
        case SDL_KEYDOWN:
          switch (event.key.keysym.sym)
          {
            case SDLK_q :
              SDL_Quit();
              return EXIT_SUCCESS;
              break;
            default :
              break;
          }
      }
  }
}










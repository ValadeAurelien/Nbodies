#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <SDL/SDL.h>

#define POT_CONST 1e-6
#define OFFSET 10
#define PI 3.141592653589793

typedef struct coord_t {float x, y, z, vx, vy, vz, m;} coord_t;
typedef enum profile_type_t {NORMAL = 0, UNIFORM = 1, EXPONANTIAL = 2} profile_type_t;
typedef enum speed_type_t {RADIAL_RANDOM = 0, WRAP_X = 1, WRAP_Z=2} speed_type_t;

typedef struct args_t 
{
  float max_X, max_Y, max_Z, 
        ray_X, ray_Y, ray_Z, 
        speed_norm, 
        min_m, max_m;
  profile_type_t profile_type;
  speed_type_t speed_type;
} args_t;
                               
__device__
float calc_ray(coord_t * coord)
{
  return  sqrtf(powf(coord->x, 2) + powf(coord->y, 2) + powf(coord->z, 2));
}

__device__
float calc_dist(coord_t * coord1, coord_t * coord2)
{
  return  sqrtf(powf(coord1->x - coord2->x, 2) + powf(coord1->y - coord2->y, 2) + powf(coord1->z - coord2->z, 2));
}

__device__
float sq(float val){return pow(val,2);}

__device__
float sym_exp(float a, float A)
{
  if (a<0) return expf(a/A);
  else return expf(-a/A);
}

__device__
coord_t potential(coord_t contrib, coord_t * a, coord_t * b)
{
  float dist = calc_dist(a, b);
  float val = POT_CONST * b->m / powf(dist, 3);
  contrib.x += val*(b->x - a->x);
  contrib.y += val*(b->y - a->y);
  contrib.z += val*(b->z - a->z);
  return contrib;
}

__global__ 
void init_coordinates_n_masses(coord_t * bodies_1, coord_t * bodies_2, curandState_t * states, size_t X, 
                               args_t args, int seed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  curand_init(seed, j*X+i, OFFSET, &states[j*X+i]);
  switch (args.profile_type)
  {
    case NORMAL :
      bodies_1[j*X+i].x = curand_normal(&states[j*X+i])*args.ray_X+args.max_X/2;
      bodies_1[j*X+i].y = curand_normal(&states[j*X+i])*args.ray_Y+args.max_Y/2;
      bodies_1[j*X+i].z = curand_normal(&states[j*X+i])*args.ray_Z+args.max_Z/2;
      break;
    case UNIFORM :
      bodies_1[j*X+i].x = curand_uniform(&states[j*X+i])*args.ray_X+args.max_X/2;
      bodies_1[j*X+i].y = curand_uniform(&states[j*X+i])*args.ray_Y+args.max_Y/2;
      bodies_1[j*X+i].z = curand_uniform(&states[j*X+i])*args.ray_Z+args.max_Z/2;
      break;
    case EXPONANTIAL :
      bodies_1[j*X+i].x = sym_exp(2*curand_uniform(&states[j*X+i])-1, args.ray_X) + args.max_X/2;
      bodies_1[j*X+i].y = sym_exp(2*curand_uniform(&states[j*X+i])-1, args.ray_Y) + args.max_Y/2;
      bodies_1[j*X+i].z = sym_exp(2*curand_uniform(&states[j*X+i])-1, args.ray_Z) + args.max_Z/2;
      break;
    default :
      break;
  }
  switch (args.speed_type)
  {
    case RADIAL_RANDOM : 
      float a = bodies_1[j*X+i].x,
            b = bodies_1[j*X+i].y,
            c = bodies_1[j*X+i].z;
      float vz = .5* args.speed_norm*(2*curand_uniform(&states[j*X+i])-1);
      float vy = -( b*c*vz + sqrtf(sq(b*c*vz)-sq((sq(a)+sq(b))*vz)+(sq(a)+sq(b))*sq(a*args.speed_norm))) / (sq(a)+sq(b));
      float vx = -(b*vy+c*vz)/a;
      bodies_1[j*X+i].vx = vx;
      bodies_1[j*X+i].vy = vy;
      bodies_1[j*X+i].vz = vz;
      break;
    case WRAP_X :
      bodies_1[j*X+i].vx = 0;
      bodies_1[j*X+i].vy = args.speed_norm*(-bodies_1[j*X+i].z/args.max_Z + 1./2);
      bodies_1[j*X+i].vz = args.speed_norm*(bodies_1[j*X+i].y/args.max_Y - 1./2);
      break;
    case WRAP_Z :
      bodies_1[j*X+i].vz = 0;
      bodies_1[j*X+i].vy = args.speed_norm*(bodies_1[j*X+i].x/args.max_X - 1./2);
      bodies_1[j*X+i].vx = args.speed_norm*(-bodies_1[j*X+i].y/args.max_Y + 1./2);
      break;
    default :
      break;
  }
  //float args.ray = calc_args.ray(&bodies_1[j*X+i]); 
  bodies_1[j*X+i].m = (args.min_m+args.max_m)/2 + curand_uniform(&states[j*X+i]) * (args.max_m-args.min_m)/2;
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
        contrib = potential(contrib, &bodies_in[j*X+i], &bodies_in[h*X+w]);
  bodies_out[j*X+i].vx = bodies_in[j*X+i].vx + contrib.x;
  bodies_out[j*X+i].vy = bodies_in[j*X+i].vy + contrib.y;
  bodies_out[j*X+i].vz = bodies_in[j*X+i].vz + contrib.z;
  bodies_out[j*X+i].x = bodies_in[j*X+i].vx + bodies_in[j*X+i].x ;
  bodies_out[j*X+i].y = bodies_in[j*X+i].vy + bodies_in[j*X+i].y ;
  bodies_out[j*X+i].z = bodies_in[j*X+i].vz + bodies_in[j*X+i].z ;
}

void update_pixels(coord_t * h_bodies, Uint32 * pixels, float max_X, float max_Y, float min_m, float max_m, 
                   size_t X, size_t Y, size_t nb_pts_X, size_t nb_pts_Y)
{
  for (int t=0; t<nb_pts_X*nb_pts_Y; t++)
      pixels[t] = 0;

  size_t px, py;
  float t;
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++)
    {
      px = ceil( nb_pts_X * h_bodies[y*X+x].x / max_X);
      py = ceil( nb_pts_Y * h_bodies[y*X+x].y / max_Y);
      t = h_bodies[y*X+x].m / (max_m - min_m) - min_m;
      if (0<py && py<nb_pts_Y && 0<px && px< nb_pts_X)
        pixels[py*nb_pts_X+px] = floor( 16711680 * t + (1-t) * 65280);
    }
}

int main(int argc, char * argv[])
{
  if (argc != 2)
  {
    printf("Enter a formated arguments file\n");
    return EXIT_SUCCESS;
  }
  
  FILE *argfile = fopen(argv[1], "r");
  if (argfile == NULL) 
  {
    printf("The argfile cannot be read...\n");
    return EXIT_FAILURE;
  }
  
  char argname [50],
       comentary[50];
  size_t N, nb_pts_X, nb_pts_Y;
  args_t args;


  fscanf(argfile, "%s", argname); fscanf(argfile, "%d", &N); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%d", &nb_pts_X); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%d", &nb_pts_Y); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.max_X); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.max_Y); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.max_Z); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.ray_X); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.ray_Y); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.ray_Z); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%d", &args.profile_type); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.speed_norm); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%d", &args.speed_type); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.min_m); fscanf(argfile, "%s", comentary);
  fscanf(argfile, "%s", argname); fscanf(argfile, "%f", &args.max_m); fscanf(argfile, "%s", comentary);
  
  printf("N = %d \n", N);
  printf("nb_pts_X = %d \n", nb_pts_X);
  printf("nb_pts_Y = %d \n", nb_pts_Y);
  printf("max_X = %f \n", args.max_X);
  printf("max_Y = %f \n", args.max_Y);
  printf("max_Z = %f \n", args.max_Z);
  printf("ray_X = %f \n", args.ray_X);
  printf("ray_Y = %f \n", args.ray_Y);
  printf("ray_Z = %f \n", args.ray_Z);
  printf("profile_type = %d \n", args.profile_type);
  printf("speed_norm = %f \n", args.speed_norm);
  printf("speed_type = %d \n", args.speed_type);
  printf("min_m  = %f \n", args.min_m);
  printf("max_m  = %f \n", args.max_m);

  fclose(argfile);

  size_t X, Y; 
  X = ceil(sqrt(N));
  Y = ceil((float) N/X);
  
  size_t blockside;
  if (N<=1024) blockside = 16;
  if ((N>1024) && (N<=4096)) blockside = 32;
  if (N>4096) blockside = 64;

  printf("X, Y, blockside = %d, %d, %d\n", X, Y, blockside);
    
  dim3 blockSize (blockside,blockside);
  dim3 gridSize (ceil( (float) X/blockside), ceil( (float) Y/blockside));

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
                                                     args, time(NULL));
  cudaMemcpy(h_bodies, d_bodies_1, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
  update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, args.max_X, args.max_Y, 
                args.min_m, args.max_m, X, Y, nb_pts_X, nb_pts_Y);
  SDL_Flip(SDL_img);

  while (true)
  {
    update_bodies<<<gridSize, blockSize>>>(d_bodies_1, d_bodies_2, X, Y);
    cudaMemcpy(h_bodies, d_bodies_2, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
    update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, args.max_X, args.max_Y, 
                  args.min_m, args.max_m, X, Y, nb_pts_X, nb_pts_Y);
    SDL_Flip(SDL_img);

    update_bodies<<<gridSize, blockSize>>>(d_bodies_2, d_bodies_1, X, Y);
    cudaMemcpy(h_bodies, d_bodies_1, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
    update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, args.max_X, args.max_Y, 
                  args.min_m, args.max_m, X, Y, nb_pts_X, nb_pts_Y);
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
            case SDLK_r :
              init_coordinates_n_masses<<<gridSize, blockSize>>>(d_bodies_1, d_bodies_2, states, X,
                                                                 args, time(NULL));
              cudaMemcpy(h_bodies, d_bodies_1, X*Y*sizeof(coord_t), cudaMemcpyDeviceToHost);
              update_pixels(h_bodies, (Uint32 *) SDL_img->pixels, args.max_X, args.max_Y, 
                            args.min_m, args.max_m, X, Y, nb_pts_X, nb_pts_Y);
              SDL_Flip(SDL_img);
              break;
            default :
              break;
          }
      }
  }
}


  //for (int i=0; i<X*Y; i++) printf("%f %f %f | ", h_bodies[i].x, h_bodies[i].y, h_bodies[i].m); printf("\n");








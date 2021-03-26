#include <cstdio>
#include <cstdlib>
#include <cstring>

const int MAXN = 2001;
const int MAXD = 150;
const int MINK = 4;
const int MAXK = 14;

char X1[MAXN][MAXD]; 
char X2[MAXN][MAXD];

void kernel_pairwise(char *a, char *b, double *K, double alpha)
{
    int n, m;
    for (n = 0; a[n] != 0; n++)
        ;
    for (m = 0; b[m] != 0; m++)
        ;
    n += 1;
    m += 1;

    double B[n][m];
    double prevB[n][m];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            B[i][j] = 1;

    for (int k = 1; k < MAXK; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                prevB[i][j] = B[i][j];
                B[i][j] = 0;
            }
        }
        for (int i = k; i < n; i++)
        {
            for (int j = k; j < m; j++)
            {
                B[i][j] = alpha * B[i - 1][j] + alpha * B[i][j - 1] - (alpha * alpha) * B[i - 1][j - 1] +
                          (a[i - 1] == b[j - 1]) * (alpha * alpha) * prevB[i - 1][j - 1];
            }
        }

        double res = 0;
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = 0; j < m - 1; j++)
            {
                if (a[i] == b[j])
                {
                    res += B[i][j];
                }
            }
        }
        K[k + 1] = alpha * alpha * res;
    }
}

void skipline(FILE *fptr)
{
    char c;
    fscanf(fptr, "%c", &c);
    while (c != '\n')
    {
        fscanf(fptr, "%c", &c);
    }
}

int readfile(char *path, char X[MAXN][MAXD]) {
    int N = 0;
    FILE *fptr;
    fptr = fopen(path, "r");
    skipline(fptr);
    while (true) {
        int x, r;
        r = fscanf(fptr, "%d,%s", &x, X[N]);
        if (r == EOF)
        {
            break;
        }
        N++;
    }
    fclose(fptr);
    return N;
}

int main(int argc, char **argv)
{
    bool test = false;
    bool diag = false;
    if (strcmp(argv[1], "test") == 0) {
        test = true;
    }
    if (strcmp(argv[1], "diag") == 0) {
        diag = true;
    }

    char *auxpath;
    double alpha;
    int p, q, M;
    int N = readfile(argv[2], X1);
    if (test) {
        M = readfile(argv[3], X2);
        auxpath = argv[4];
        alpha = strtod(argv[5], NULL);
        if (!diag) {
            p = atoi(argv[6]);
            q = atoi(argv[7]);
        }
    } else {
        M = N;
        auxpath = argv[3];
        alpha = strtod(argv[4], NULL);
        if (!diag) {
            p = atoi(argv[5]);
            q = atoi(argv[6]);
        }
    }
    if (diag) {
        p = 0;
        q = N;
    }

    int PATHLEN = strlen(auxpath);
    FILE *fs[MAXK + 1];
    for (int k = MINK; k <= MAXK; k++)
    {
        char out_path[100];
        strcpy(out_path, auxpath);
        sprintf(out_path + PATHLEN, "%d", k);
        strcat(out_path, ".csv");
        fs[k] = fopen(out_path, "w");
    }


    for (int i = p; i < q; i++)
    {
        if (p == 0 && !diag)
        {
            fprintf(stderr, "%d ", i);
        }

        if (diag) {
            double K[MAXK+1];
            for (int k = 0; k <= MAXK; k++) {
                K[k] = 0;
            }
            kernel_pairwise(X1[i], X1[i], K, alpha);
            for (int k = MINK; k <= MAXK; k++)
            {
                fprintf(fs[k], "%.14f", K[k]);
                if (i < M - 1)
                    fprintf(fs[k], ",");
            }
        } else {
            for (int j = 0; j < M; j++)
            {
                double K[MAXK+1];
                for (int k = 0; k <= MAXK; k++) {
                    K[k] = 0;
                }

                if (j >= i || test) {
                    if (test)
                       kernel_pairwise(X1[i], X2[j], K, alpha);
                    else
                        kernel_pairwise(X1[i], X1[j], K, alpha);
                }
                
                for (int k = MINK; k <= MAXK; k++)
                {
                    fprintf(fs[k], "%.14f", K[k]);
                    if (j < M - 1)
                        fprintf(fs[k], ",");
                    else
                        fprintf(fs[k], "\n");
                }
            }
        }
    }
    fprintf(stderr, "\n");

    for (int k = MINK; k <= MAXK; k++)
    {
        fclose(fs[k]);
    }
}

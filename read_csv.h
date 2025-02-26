#ifndef READ_CSV_H
#define READ_CSV_H
#include <stdbool.h>
#include <stddef.h>

// this is just MVP for testing
// doesnt handle empty values
// only reads numbers

#define MAX_COLS 1024
#define MAX_ROWS 50000
#define MAX_LINE_SIZE 2048
#define MAX_COL_SIZE 128

typedef struct
{
    float *data[MAX_ROWS];
    char *col_names[MAX_COLS];
    size_t n_rows;
    size_t n_cols;
} csv;

csv csv_read(const char *file_name, const char *delim, bool includes_col_names);
void csv_free(csv csv_data);

#endif
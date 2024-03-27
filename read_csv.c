#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "read_csv.h"


csv csv_read(const char *file_name, const char *delim){
    FILE *csv_file = fopen(file_name, "r"); //open file with read mode

    csv csv_data;
    
    char line[MAX_LINE_SIZE];
    size_t col_count = 0;
    size_t row_count = 0;
    size_t cur_col_count = 0;

    //append \n to delim
    char enddelim[sizeof(*delim) + 3];
    strcpy(enddelim, delim);
    enddelim[sizeof(*delim) +1] = '\n';
    enddelim[sizeof(*delim) +2] = '\0';
    

    while(fgets(line, MAX_LINE_SIZE, csv_file)){//read line by line
        if(row_count == 0){
            for(char* tok = strtok(line, delim); tok && *tok; tok = strtok(NULL, enddelim)){
                col_count++; //use the colnames row(row_count == 0) to count rows. stupid but MVP
            }
            csv_data.n_cols = col_count;

        } else if (row_count < MAX_ROWS){
            float *column_pointer = (float*)malloc(col_count * sizeof(float));
            for(char* tok = strtok(line, delim); tok && *tok; tok = strtok(NULL, ";\n")){
                if(cur_col_count >= col_count){ //if a row has too many columns there is an error
                    printf("ERROR reading csv (index error)");
                    exit(EXIT_FAILURE);
                } 

                float value = strtof(tok, NULL); //converts string value to float
                column_pointer[cur_col_count] = value; //adds the value into the allocated memory
                cur_col_count++;
            }
            csv_data.data[row_count - 1] = column_pointer;
            csv_data.n_rows = row_count;
        }
        

        cur_col_count = 0;
        row_count++;
        //split on token ; and sav in column
        //then save/malloc all the columns the columns_in_row
        //lastly malloc the row pointers into the array_of_line_pointers
    }
    return csv_data;
}

void csv_free(csv csv_data){
    for(size_t i = 0; i < csv_data.n_rows; i++){
        free(csv_data.data[i]);
    }
}


/*int main(){
    
    csv data = read_csv("test.csv", ";");

    for(size_t i = 0; i < data.n_rows; i++){
        for (size_t j = 0; j < data.n_cols; j++)
        {
            printf("%zu, %zu: %f ", i, j, data.data[i][j]);
        }
        printf("\n");
    }

    return 0;
}*/
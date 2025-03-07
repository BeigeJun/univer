#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main() {
	FILE* fp;
	char str1[100], arr[10];
	char* token;
	int adress = 0;
	fp = fopen("a.txt", "r");
	if (fp == NULL) {
		printf("파일 읽기모드 열기에 실패");
		return -1;
	}

	while (!feof(fp)) {
		fgets(str1, 100, fp); 
		token = strtok(str1, " ,"); 
			if (strcmp(token, "MOV") == 0 || strcmp(token, "CMP") == 0 
				|| strcmp(token, "SUB") == 0 || strcmp(token, "ADD") == 0) {
				printf("%s[MOV,CMP,SUB,ADD] ", token);
				
				token = strtok(NULL, " ,");

				if (strcmp(token, "AX") == 0 || strcmp(token, "BX") == 0 
					|| strcmp(token, "CX") == 0 || strcmp(token, "DX") == 0) {
					printf("%s[REG]", token);
				}
				else {
					printf("%s[SYMBOL]", token);
				}

				
				token = strtok(NULL," ,");
				arr[0] = token[0]; 
				if (strcmp(token, "AX") == 0 || strcmp(token, "BX") == 0 
					|| strcmp(token, "CX") == 0 || strcmp(token, "DX") == 0
					|| strcmp(token, "AX\n") == 0 || strcmp(token, "BX\n") == 0
					|| strcmp(token, "CX\n") == 0 || strcmp(token, "DX\n") == 0) {
					printf("%s[REG]\n", token);
					printf("상대적 주소 : %d ~ ", adress);
					adress = adress + 2;
					printf("%d\n", adress);
				}

				else if (isdigit(arr[0])) {
					printf("%s[NUM]\n", token);
					printf("상대적 주소 : %d ~ ", adress);
					adress = adress + 3;
					printf("%d\n", adress);
				}
				else {
					printf("%s[SYMBOL]\n", token);
					printf("상대적 주소 : %d ~ ", adress);
					adress = adress + 4;
					printf("%d\n", adress);
				}


			}
			else if (strcmp(token, "JMP") == 0) {
				printf("%s[JMP] ", token);
				token = strtok(NULL, " ,");
				printf("%s\n", token);
				printf("상대적 주소 : %d ~ ", adress);
				adress = adress + 2;
				printf("%d\n", adress);
			}
			else if(strcmp(token, "JX") == 0) {
				printf("%s[JX] ", token);
				token = strtok(NULL, " ,");
				printf("%s\n", token);
				printf("상대적 주소 : %d ~ ", adress);
				adress = adress + 2;
				printf("%d\n", adress);
			}
			else if (strcmp(token, "INT") == 0) {
				printf("%s[INT] ", token);
				token = strtok(NULL, " ,");
				printf("%s\n", token);
				printf("상대적 주소 : %d ~ ", adress);
				adress = adress + 2;
				printf("%d\n", adress);
			}
			else {
				printf("%s[NOTTHING]\n", token);
			}
	}
	fclose(fp);
	return 0;
}

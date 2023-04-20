#include <Windows.h>
#include <tchar.h>
#include "resource.h"

#define lengthMAX 256

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, 
			WPARAM wParam, LPARAM lParam);

LPCTSTR lpszClass = TEXT("Mid-Exam");			

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, //WINAPI : 윈도우 프로그램이라는 의미
		   LPSTR lpszCmdLine, int nCmdShow)						 //hInstance : 운영체제의 커널이 응용 프로그램에 부여한 ID
{																 //szCmdLine : 커멘트라인 상에서 프로그램 구동 시 전달된 문자열
	HWND	hwnd;												 //iCmdShow : 윈도우가 화면에 출력될 형태
	MSG		msg;

	WNDCLASS WndClass;													 //WndClass 라는 구조체 정의									 
	WndClass.style			= CS_SAVEBITS;								 //출력스타일 : 수직/수평의 변화시 다시 그림
	WndClass.lpfnWndProc	= WndProc;									 //프로시저 함수명	
	WndClass.cbClsExtra		= 0;										 //O/S 사용 여분 메모리 (Class)
	WndClass.cbWndExtra		= 0;										 //O/s 사용 여분 메모리 (Window)
	WndClass.hInstance		= hInstance;								 //응용 프로그램 ID
	WndClass.hIcon			= LoadIcon(NULL, IDI_APPLICATION);			 //아이콘 유형
	WndClass.hCursor		= LoadCursor(NULL, IDC_ARROW);				 //커서 유형
	WndClass.hbrBackground	= (HBRUSH)GetStockObject(WHITE_BRUSH);	   	 //배경색   
	WndClass.lpszMenuName	= MAKEINTRESOURCE(IDR_MENU1);										 //메뉴 이름*/
	WndClass.lpszClassName	= lpszClass;								 //클래스 이름
	
	RegisterClass(&WndClass);		//앞서 정의한 윈도우 클래스의 주소 (윈도우 클래스를 커널에 등록시킴)

	hwnd = CreateWindow(lpszClass,								 //윈도우가 생성되면 핸들(hwnd)이 반환 (윈도우 생성)
		lpszClass,												 //윈도우 클래스, 타이틀 이름
		WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_HSCROLL,									 //윈도우 스타일
		200,													 //윈도우 위치, x좌표
		100,				    								//윈도우 위치, y좌표
		600,													 //윈도우 폭   
		400,													 //윈도우 높이   
		NULL,													 //부모 윈도우 핸들	 
		NULL,													 //메뉴 핸들
		hInstance,    											 //응용 프로그램 ID
		NULL     												 //생성된 윈도우 정보
		);

	ShowWindow(hwnd, nCmdShow);									 // (윈도우 정보 전달 및 윈도우 출력)


	while(GetMessage(&msg, NULL, 0, 0))							 //WinProc()에서 PostQuitMessage() 호출 때까지 처리 (윈도우 메시지 루프)
	{															 // 
		TranslateMessage(&msg);
		DispatchMessage(&msg);									 //WinMain -> WinProc  
	}
	return (int)msg.wParam;
}


HANDLE fileHandle;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {

	static TCHAR linebfr[256][256] = {0, };
	static int i, row, col;
    static COLORREF textColor = RGB(0, 0, 0);
	static int line, len;
	static int block;
    switch (msg) {
	case WM_CREATE:
		row = 0; 
		col = 0;
		break;
	case WM_COMMAND:
		if(wParam == ID_fileOpen){
			fileHandle=CreateFile(TEXT("a.txt"), GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
			row = 0;
			col = 0;
			block = 0;
			//CloseHandle(fileHandle);
			break;
		}
		if(wParam == ID_fileClose){
			CloseHandle(fileHandle);
			//PostQuitMessage(0);
		}
		InvalidateRect(hwnd, NULL, TRUE);
		break;
	case WM_CHAR:
		
		if(wParam == VK_RETURN){
			TCHAR strRead[256]={'\0',};
			DWORD dwBytes=0;
			DWORD dwRead;
			WORD wc= 0xFEFF;
			int index = 0;
			//fileHandle=CreateFile(TEXT("a.txt"), GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
			strRead[index++] = wc;
			for(int i=0; i<=row; i++){
				for(int j=0; linebfr[i][j] != NULL; j++){
					strRead[index++] = linebfr[i][j];
				}
				if(i != row){
					strRead[index++] = '\n';
				}
			}
			WriteFile(fileHandle, strRead, index*2, &dwRead, NULL);

			//CloseHandle(fileHandle);
			row++;
			col = 0;
		}
		else {
			linebfr[row][col++] = wParam;
			linebfr[row][col] = '\0';
		}
		
		InvalidateRect(hwnd, NULL, TRUE);
		break;

    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

			for(i=0; i<=256; i++)
				TextOut(hdc,0,0+20*i,linebfr[i],lstrlen(linebfr[i]));
			
            EndPaint(hwnd, &ps);
        }
        break;

    case WM_DESTROY:
		CloseHandle(fileHandle);
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }

    return 0;
}

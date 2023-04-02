#include <windows.h>
#include <WindowsX.h>
#include <math.h>
#include <stdlib.h>
#define BSIZE 20



LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, 
         WPARAM wParam, LPARAM lParam);

LPCTSTR lpszClass = TEXT("dkdkdkdkddkdkkkdk");

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, //WINAPI : 윈도우 프로그램이라는 의미
         LPSTR lpszCmdLine, int nCmdShow)                   //hInstance : 운영체제의 커널이 응용 프로그램에 부여한 ID
{                                                 //szCmdLine : 커멘트라인 상에서 프로그램 구동 시 전달된 문자열
   HWND   hwnd;                                     //iCmdShow : 윈도우가 화면에 출력될 형태
   MSG      msg;
   WNDCLASS WndClass;                                  //WndClass 라는 구조체 정의                            
   WndClass.style         = CS_HREDRAW | CS_VREDRAW;          //출력스타일 : 수직/수평의 변화시 다시 그림
   WndClass.lpfnWndProc   = WndProc;                      //프로시저 함수명
   WndClass.cbClsExtra      = 0;                         //O/S 사용 여분 메모리 (Class)
   WndClass.cbWndExtra      = 0;                         //O/s 사용 여분 메모리 (Window)
   WndClass.hInstance      = hInstance;                   //응용 프로그램 ID
   WndClass.hIcon         = LoadIcon(NULL, IDI_APPLICATION);    //아이콘 유형
   WndClass.hCursor      = LoadCursor(NULL, IDC_ARROW);       //커서 유형
   WndClass.hbrBackground   = (HBRUSH)GetStockObject(WHITE_BRUSH);//배경색   
   WndClass.lpszMenuName   = NULL;                         //메뉴 이름
   WndClass.lpszClassName   = lpszClass;                   //클래스 이름
   RegisterClass(&WndClass);                            //앞서 정의한 윈도우 클래스의 주소

   hwnd = CreateWindow(lpszClass,                         //윈도우가 생성되면 핸들(hwnd)이 반환
      lpszClass,                                     //윈도우 클래스, 타이틀 이름
      WS_OVERLAPPEDWINDOW,                            //윈도우 스타일
      100,                                  //윈도우 위치, x좌표
      50,                                  //윈도우 위치, y좌표
      620,                                  //윈도우 폭   
      620,                                  //윈도우 높이   
      NULL,                                        //부모 윈도우 핸들    
      NULL,                                        //메뉴 핸들
      hInstance,                                      //응용 프로그램 ID
      NULL                                          //생성된 윈도우 정보
   );
   ShowWindow(hwnd, nCmdShow);                            //윈도우의 화면 출력
   UpdateWindow(hwnd);                                  //O/S 에 WM_PAINT 메시지 전송

   while(GetMessage(&msg, NULL, 0, 0))                      //WinProc()에서 PostQuitMessage() 호출 때까지 처리
   {
      TranslateMessage(&msg);
      DispatchMessage(&msg);                            //WinMain -> WinProc  
   }
   return (int)msg.wParam;
}

HDC hdc;

int Length(int x1, int y1) {
   if(x1>=y1){
      return x1=y1;
   }
   else{
      return y1-x1;
   }
   //피타고라스 정의를 이용하여 원의 중심으로부터 마우스포인터까지 거리를 반환
}

bool correct(int x, int y, int x1, int y1) {
   if (Length(x, x1) < BSIZE&&Length(y,y1)<BSIZE) {
      return TRUE;
   }
   else {
      return FALSE;
   }
}
static int x = 0, y = 0; bool check = FALSE;
void CALLBACK TimerProc(HWND hwnd, UINT uMsg, UINT idEvent, DWORD dwTime)
{
	HDC hdc;
	int i;
	hdc=GetDC(hwnd);
	for(int i = 0 ; i < 10 ; i ++)
	{

		x=rand()%600;
		y=rand()%600;
		TextOut(hdc,x,y,"H",1);
	}
		ReleaseDC(hwnd, hdc);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)     
{

   PAINTSTRUCT ps;
   static bool check;
   static int x1,y1;

   switch (iMsg) 
   {
   case WM_CREATE:
     check=FALSE;
	 for(int i = 0 ; i < 10 ; i ++)
	 {
		 x=rand()%600;
		 y=rand()%600;
		 TextOut(hdc,x,y,"H",1);
	 }
     break;




   case WM_PAINT:
      hdc = BeginPaint(hwnd, &ps);
      Rectangle(hdc,x1-7,y1-7,x1+7,y1+7);
      if(check==TRUE){
         SetTextColor(hdc, RGB(255, 0, 0));
		 TextOut(hdc,x,y,"H",1);
		 TextOut(hdc,550,550,"Succes",6);
		 check = FALSE;
		 EndPaint(hwnd,&ps);
         break;
      }
	  TextOut(hdc,x,y,"H",1);
      TextOut(hdc,550,550,"FALSE ",6);
      EndPaint(hwnd,&ps);
      break;

   

   case WM_LBUTTONDOWN:
      x1=LOWORD(lParam);
      y1=HIWORD(lParam);

      
      if(correct(x,y,x1,y1)){
      check=TRUE;
      }
	  else
		  check = FALSE;
      InvalidateRect(hwnd, NULL,TRUE);
      break;
    /*  
   case WM_LBUTTONUP:
      check=FALSE;
      InvalidateRect(hwnd, NULL, TRUE);
      break;
	  */
   

  

   case WM_DESTROY:
      PostQuitMessage(0);
      break;
   } 
   return DefWindowProc(hwnd, iMsg, wParam, lParam);          //CASE에서 정의되지 않은 메시지는 커널이 처리하도록 메시지 전달
}

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
const int N = 1e5;
int n;
int a[N];
int b[N];
int W(int pos);
int main() {
   cin>>n;
   for(int i = 0; i < n+1;i++){
   	cin>>a[i];
   }
   b[0] = 0;
   for(int i = 1; i < n+1;i++){
   	cin>>b[i];
   }
   for(int i = 1;i <= n;i++){
   	  int w = W(i);
   	  cout<<w;
   }
}
int W(int pos)
{
   int key = b[pos];
   b[pos] = 0;
   int power = 0;
   int power_need = 0;
   
   for(int i = 0;i <= n;i++){
   	    power += (b[i] - a[i]);
   	    if(power < 0){
   	   	  power_need += abs(power);
   	   	  power = 0;
		  }
   }
   b[pos] = key;
   return power_need;	
}

#include <iostream>
#include <fstream>
using namespace std;
int main()
{
      std::ofstream myfile;
      myfile.open("stuff.dat");
      myfile << "This is the first cell in the first column.\n";
      myfile << "a,b,c,\n";
      myfile << "c,s,v,\n";
      myfile << "1,2,3.456\n";
      myfile << "semi;colon";
      myfile.close();
      return 0;
}
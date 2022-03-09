// write_csv.cpp
// updated 8/12/16

// the purpose of this code is to demonstrate how to write data
// to a csv file in C++

// inlcude iostream and string libraries
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// create an ofstream for the file output (see the link on streams for
// more info)
ofstream outputFile;
ofstream fs;
// create a name for the file output
std::string filename = "exampleOutput.csv";

// create some variables for demonstration
int i;
int A;
int B;
int C;

int main()
{
    // create and open the .csv file
    fs.open(outputFile,filename);
    
    // write the file headers
    outputFile << "Column A" << "," << "Column B" << "Column C" << std::endl;
    
    // i is just a variable to make numbers with in the file
    i = 1;
  
    // write data to the file
    for (int counter = 0; counter <  10; counter++)
    {
        // make some data
        A = i + 5;
        B = i + 10;
        C = i + 20;
        
        // write the data to the output file
        outputFile << A << "," << B << "," << C << std::endl;
        
        // increase i
        i = i * 5;
    }
    
    // close the output file
    outputFile.close();
    return 0;   
}

//#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream> // 用于读写存储在内存中的string对象

int main(void)
{
    // 制作CSV文件
    /* Name,age,height
     * Tom,21,172.8
     * John,25,189.5
     */
     std::ofstream outFile;
     outFile.open("./csvTest.csv", std::ios::out);
     outFile << "Name" << "," << "age" << "," << "height" << std::endl;
     outFile << "Tom" << "," << 21 << "," << 172.8 << std::endl;
     outFile << "John" << "," << 25 << "," << 189.5 << std::endl;
     outFile.close();

     // 读取CSV文件
     struct CSVDATA {
      std::string name;
      int age;
      double height;
     };
     std::ifstream inFile("./csvTest.csv", std::ios::in);
     std::string lineStr;
     std::vector<struct CSVDATA> csvData;
     std::getline(inFile, lineStr); // 跳过第一行(非数据区域)
     while (std::getline(inFile, lineStr)) {
          std::stringstream ss(lineStr); // string数据流化
          std::string str;
          struct CSVDATA csvdata;
          std::getline(ss, str, ','); // 获取 Name
          csvdata.name = str;
          std::getline(ss, str, ','); // 获取 age
          csvdata.age = std::stoi(str);
          std::getline(ss, str, ','); // 获取 height
          csvdata.height = std::stod(str);

          csvData.push_back(csvdata);
     }
     // 显示读取的数据
     for (int i = 0; i < csvData.size(); i ) {
          std::cout << csvData[i].name << "," << csvData[i].age << "," << csvData[i].height << std::endl;
     }

     getchar();
     return 0;    
}
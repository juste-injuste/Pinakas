#include <iostream>
#include <fstream>
#include <chrono>
#include "matrix.hpp"
#include "mudelizer.hpp"

#define tic	auto _start = std::chrono::high_resolution_clock::now()	
#define toc	auto _stop = std::chrono::high_resolution_clock::now(); std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start).count()*0.000001 << " s\n"

using std::cout;
using std::endl;

void program(void) {
	std::ifstream    file;
	std::string      line;
	int nlines = 0;
	file.open("data.csv");
	for (int i=0; !file.eof(); i++) {
		file >> line;
		nlines++;
	}
	nlines--;
	matrix xdata(1, nlines), ydata(1, nlines);
	file.clear();
	file.seekg(0);
	for (int i=0, j; i<nlines; i++) {
		file >> line;
		for (j=1; j<line.size(); j++) if (line[j]==',') break;
		xdata(0, i) = std::stod(&line[0]);
		ydata(0, i) = std::stod(&line[j+1]);
	}
	file.close();
	
	cout << CLEARSCREEN << "starting" << endl;
	
	tic;
		linearize(xdata, ydata)[0].show();
		linearize(xdata, ydata)[1].show();

		resample(ydata, 3).show();
	toc;

	return;
}

int main() {
	std::cout << CLEARSCREEN;
	program();
	
	return 0;
}

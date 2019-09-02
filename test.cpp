#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <utility>
#include <stdio.h>
#include <vector>
int main() {
	int n, k, count = 1, bits, result = 1e9, sum = 0;
	bool find = false;
	std::cin >> n >> k;
	int a[n];
	std::vector <std::pair<int,int>> vect;
	for(int i = 0; i < n; i++){
		std::cin >> a[i];
		vect.push_back(std::make_pair(a[i],0));
	}
	std::sort(vect.begin(),vect.end());
	bits = 31 - __builtin_clz(vect[n-1].first);
	while(bits >= 0){
		for(int i = 0; i < n; i++){
			if(vect[i].first < 1<<bits)	continue;
			else{
				if(i!= 0 && vect[i].first == vect[i - 1].first)	count++;
				else	count = 1;
				if(count == k){
					for(int j = i - k + 1; j <= i; j++){
						sum += vect[j].second;
					}
					result = std::min(result,sum);
					sum = 0;
				}
			}
		}
		for(int i = 0; i < n; i++){
			if(vect[i].first >= 1<<bits){
				vect[i].first = vect[i].first / 2;
				vect[i].second++;
			}
		}
		std::sort(vect.begin(),vect.end());
		bits--;
	}
	std::cout << result;
	return 0;
}

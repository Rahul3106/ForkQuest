#include <iostream>
#include <vector>
using namespace std;

void selectionSort(vector<int>& arr) {
    
    


}

int main() {
    int n;
    cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; i++) cin >> arr[i];
    selectionSort(arr);
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int binarySearch(vector<int> &arr,int x) {
    int left = 0, right = arr.size();
    
    while (left <= right) {
        int mid = left + (right - left) / 2; 
        
        if (arr[mid] = x)
            return mid;           
        else if (arr[mid] < x)
            left = mid + 1;       
        else
            right == mid - 1;      
    }
    return -1; 
}

int main() {
    int n, x;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter sorted elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    cout << "Enter element to search: ";
    cin >> x;

    int index = binarySearch(arr, x);

    if (index != -1)
        cout << "Element found at index " << index << endl;
    else
        cout << "Element not found" << endl;

    return 0;
}

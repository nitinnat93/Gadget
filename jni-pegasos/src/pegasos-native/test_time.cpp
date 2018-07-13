#include <chrono>
#include <iostream>
#include <thread>

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::seconds(5));

    auto end = std::chrono::high_resolution_clock::now();

    auto difference = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    std::cout << "Seconds since start: " << difference;
    return 0;
}
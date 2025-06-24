#include <iostream>
#include <iomanip>
using namespace std;

/*各月の日数を取得*/
int getDaysInMonth(int year, int month) {
    if (month == 2) {
        /*閏年の判定*/
        bool isLeap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        return isLeap ? 29 : 28;
    }
    if (month == 4 || month == 6 || month == 9 || month == 11)
        return 30;
    return 31;
}

/*月初めの曜日をを求める*/
int getStartWeekday(int year, int month) {
    if (month < 3) {
        month += 12;
        year -= 1;
    }
    int y = year;
    int m = month;
    int d = 1;
    /*ツェラ―の公式*/
    int w = (d + 2*m + 3*(m+1)/5 + y + y/4 - y/100 + y/400) % 7;
    return w;
}

void printCalender(int year, int month) {
    int start = getStartWeekday(year, month);
    int days = getDaysInMonth(year, month);

    cout << "\n" << year << "年 " << month << "月\n";
    cout << " 日  月  火  水  木  金  土\n";

    for (int i = 0; i < start; ++i) {
        cout << "    ";
    }
    for (int d = 1; d <= days; ++d) {
        cout << setw(3) << d << " ";
        if ((d + start) % 7 == 0) cout << endl;
    }
    cout << endl;
}


int main () {
    int year, month;
    cout << "年を入力：";
    cin >> year;
    cout << "月を入力：";
    cin >> month;
    printCalender(year, month);
}
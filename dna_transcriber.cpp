//
//  main.cpp
//  MCP_tools
//
//  Created by Olivia Gallup on 16/04/2019.
//  Copyright © 2019 Olivia Gallup. All rights reserved.
//

// DNA transcriber; turns A to T, C to G, G to C, and T to A. EZ MANI


#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <locale>

using namespace std;

string getSeq() {
    string seq;
    cout << "Copy paste in the sequence." << endl;
    cin >> seq;
    return seq;
}

bool isSpace(char c) {
    return (c == ' ' || c == '\n');
}

string removeSpaces(string s)
{
//    int i = 0, j = 0;
//    while (s[i])
//    {
//        if (s[i] != ' ')
//            s[j++] = s[i];
//        i++;
//    }
//    s[j] = '\0';
//    return s;
    
//    int i = 0;
//    char c;
//
//    s.erase(remove_if(s.begin(), s.end(), isspace), s.end());
//
//    return s;
    int currentOld = 0;
    int currentNew = 0;
    string newS;
    while(s[currentOld]){
        cout << s[currentOld] << endl;
        if(s[currentOld] != ' ') {
            newS[currentNew] = s[currentOld];
            currentNew++;
        }
        currentOld++;
    }
    return newS;
}

string transcribe(string seq) {
    int i=0;
    while(seq[i] != '\0') {
        if(seq[i] == ' ') {
            i++;
        }
        cout << seq << i << endl;
        i++;
    }
    
    return seq;
}





int main(int argc, const char * argv[]) {
    string seq = getSeq();
//    seq = removeSpaces(seq);
    cout << seq << endl;

    transcribe(seq);
    
    return 0;
}

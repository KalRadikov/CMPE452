%% 452 Assignment 3 
%
%   Description: Transforms the given sound csv values into a form of 
%   eigenvalue problem. Input is taken in the form of 2 sound sources and a
%   PCA network is then applied to find the approximation of the first 
%   principal component. The reult is one isolated sound source.
%
%   Author: Kal Radikov - 10157529
%   
%   Date: 11-17-2016
%
%   Iterations: 1
clear; close; clc 

%% Beginning of PCA network code

% Importing the data from csv file. There are 2 column in the data
% structure for each sound recording in the csv.
Data = csvread('sound.csv');
[m,n] = size(Data);
sum = [0,0];

%% Determining the Mean Vector to be subtracted from the Data

% Calculate the total sum of all values in the csv
for i = 1:size(Data,1)
    sum = [sum(1)+ Data(i,1), sum(2) + Data(i,2)];
end

% Divide the sum by the size of the data (50,000) to get the Mean Vector
meanVector = sum/(size(Data,1));

% Normalizing the original Data by subtracting the Mean Vector 
for i = 1:size(Data,1)
    Data(i,:) = [Data(i,1) - meanVector(1), Data(i,2) - meanVector(2)];
end

%% Training network and calculating the correct weights

W = [1,0]; 
dW = zeros(1, 2);
LearnRate = 0.1;

for i = 1:size(Data, 1)
    
    % Calculate 'y' variable of the network using dotproduct of weight
    % matrix and orginal Data 
    y = dot(Data(i, :), W);
    
    % Calculate 'k' variable and 'delta W' of the network
    K = y * y;
    
    % deltaW = n * (y * x - KW)
    dW = LearnRate * ((Data(i, :) * y) - (K * W));
    
    % Apply the change to the original weights. Final magnitude of weights
    % should approach the value '1'. 
    W = W + dW;
end  

%% Printing to output files

% Print output vectors to text file
fileID = fopen('Readme.txt', 'wt');
fprintf(fileID, 'The final weights are: \n');
fprintf(fileID, 'Weight 1: %3.4f \n', W(1, 1));
fprintf(fileID, 'Weight 2: %3.4f \n', W(1, 2));
fclose(fileID);

% iterate through input data using new weights to correct the audio file
combined = zeros(m, 1);
for i = 1 : m
    combined(i, 1) = dot(Data(i, :), W);
end

% write output to csv file
csvwrite('Output.csv', combined);

%import audio-rate from file
[test, Fs] = audioread('000001100mix1.wav');

%% Playing output sound 

%convert audio file into wav
audiowrite('Output.wav', combined, Fs);

sound(combined, Fs);


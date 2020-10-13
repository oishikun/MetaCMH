label = zeros(269648,81);
namelist = dir('AllLabels\*.txt');
len = length(namelist);
for j = 1:len
    filename = namelist(j).name;
    f = fullfile('AllLabels',filename);
    x = load(f);
    label(:,j) = x;
end

name = fullfile('NUS_WID_Tags', 'AllTags1k.txt');
text = load(name);

fpn = fopen('image.txt', 'rt');
image = zeros(269648,224,224,3,'uint8');
i = 1;
while feof(fpn) ~= 1
    file = fgetl(fpn);
    f = fullfile('Flickr',file);
    a = imread(f);
    b = imresize(a, [224,224]);
    if size(size(b),2) == 2
        b3(:,:,1) = b;
        b3(:,:,2) = b;
        b3(:,:,3) = b;
        b = b3;
    end
    image(i,:,:,:) = b;
    i = i + 1
end

save('NUS-wide81.mat', 'image', 'text', 'label', '-v7.3')

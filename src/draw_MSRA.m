function draw_MSRA()
 
    db_path = '..\..\Datasets\cvpr15_MSRAHandGestureDB';
    jointNum =  21;

    folder_list = ["1","2","3","4","5","6","7","8","9","I","IP","L","MP","RP","T","TIP","Y"];
    modelNum = 1;

    coord_pixel = load('test_res_pixel.txt');
    coord_pixel = reshape(coord_pixel,[size(coord_pixel,1),2,jointNum]); %frameNum, (x,y,z), jointNum
    coord_world = load('test_res.txt');
    coord_world = reshape(coord_world,[size(coord_world,1),3,jointNum]); %frameNum, (x,y,z), jointNum

    modelId = 0;
    fid = 1;
    while modelId < modelNum
        
        folderId = 1;
        while folderId <= size(folder_list,2)
            
            annot_path = strcat(db_path,'\P',num2str(modelId),'\',folder_list(folderId),'\','joint.txt');
            
            fp_gt = fopen(annot_path);
            tline = fgetl(fp_gt);

            frameNum = str2num(tline);

            frameId = 1;
            tline = fgetl(fp_gt);
            while frameId <= frameNum
    
                fprintf('%d / 76391\n',fid);
                
                img = draw_2Dskeleton_MSRA(tline,frameId,folderId,modelId,coord_pixel(fid,:,:));
                f = draw_3Dskeleton_MSRA(img,coord_world(fid,:,:));

                set(gcf, 'InvertHardCopy', 'off');
                set(gcf,'color','w');
                saveas(f, strcat('.\output\',int2str(fid)), 'jpg');
                close(f);

                fid = fid + 1;
                frameId = frameId + 1;
                tline = fgetl(fp_gt);

            end
            fclose(fp_gt);
            folderId = folderId + 1;
        end
        modelId = modelId + 1;
    end

end

function d2img = draw_2Dskeleton_MSRA(tline,frameId,folderId,modelId,coord_pixel)
    
    db_path = '..\..\Datasets\cvpr15_MSRAHandGestureDB';
    jointNum =  21;
    cubicSz = 200;
    imgWidth = 320;
    imgHeight = 240;
    folder_list = ["1","2","3","4","5","6","7","8","9","I","IP","L","MP","RP","T","TIP","Y"];
    
    coord_pixel = squeeze(coord_pixel);
    rgb_img = ones(imgHeight,imgWidth,3);
    refDepths = zeros(1,jointNum);
    line_width = 4;

    bin_name = strcat(db_path,'\P',num2str(modelId),'\',folder_list(folderId),'\',sprintf('%06d',frameId-1),'_depth.bin');
    splitted = strsplit(tline);

    for jid = 1:jointNum
        refDepths(1,jid) = -str2num(splitted{(jid-1)*3+3});
    end
    refDepth = (min(refDepths(:)) + max(refDepths(:)))/2;

    fp_bin = fopen(bin_name,'r');
    img_info = fread(fp_bin,6,'int');
    left = img_info(3);
    top = img_info(4);
    right = img_info(5)-1;
    bottom = img_info(6)-1;

    img = fread(fp_bin,[right-left+1 bottom-top+1],'float');
    img = permute(img,[2,1]);
    img(img==0) = refDepth+cubicSz/2;
    fclose(fp_bin);
    
    img = img - refDepth;
    img = img/(cubicSz/2);
    img(img>1) = 1;
    img(img<-1) = -1;
    
    img = img + 1;
    img = img/2;
    
    rgb_img(:,:,1) = 255/255;
    rgb_img(:,:,2) = 240/255;
    rgb_img(:,:,3) = 204/255;
    
    rgb_img(top:bottom,left:right,1) = img*255/255;
    rgb_img(top:bottom,left:right,2) = img*240/255;
    rgb_img(top:bottom,left:right,3) = img*204/255;
    
    
    f = figure;
    set(f, 'visible', 'off');
    imshow(rgb_img);
    hold on;

    plot([coord_pixel(1,1),coord_pixel(1,18)],[coord_pixel(2,1),coord_pixel(2,18)],'Color',[255/255,153/255,153/255],'LineWidth',line_width) %wrist to thumb mcp
    plot([coord_pixel(1,18),coord_pixel(1,19)],[coord_pixel(2,18),coord_pixel(2,19)],'Color',[255/255,102/255,102/255],'LineWidth',line_width) %thumb mcp to thumb pip
    plot([coord_pixel(1,19),coord_pixel(1,20)],[coord_pixel(2,19),coord_pixel(2,20)],'Color',[255/255,51/255,51/255],'LineWidth',line_width) %thumb pip to thumb dip
    plot([coord_pixel(1,20),coord_pixel(1,21)],[coord_pixel(2,20),coord_pixel(2,21)],'Color',[255/255,0/255,0/255],'LineWidth',line_width) %thumb dip to thumb tip

    plot([coord_pixel(1,1),coord_pixel(1,2)],[coord_pixel(2,1),coord_pixel(2,2)],'Color',[153/255,255/255,153/255],'LineWidth',line_width) %wrist to index mcp
    plot([coord_pixel(1,2),coord_pixel(1,3)],[coord_pixel(2,2),coord_pixel(2,3)],'Color',[102/255,255/255,102/255],'LineWidth',line_width) %index mcp to index pip
    plot([coord_pixel(1,3),coord_pixel(1,4)],[coord_pixel(2,3),coord_pixel(2,4)],'Color',[51/255,255/255,51/255],'LineWidth',line_width) %index pip to index dip
    plot([coord_pixel(1,4),coord_pixel(1,5)],[coord_pixel(2,4),coord_pixel(2,5)],'Color',[0/255,255/255,0/255],'LineWidth',line_width) %index dip to index tip

    plot([coord_pixel(1,1),coord_pixel(1,6)],[coord_pixel(2,1),coord_pixel(2,6)],'Color',[255/255,204/255,153/255],'LineWidth',line_width) %wrist to middle mcp
    plot([coord_pixel(1,6),coord_pixel(1,7)],[coord_pixel(2,6),coord_pixel(2,7)],'Color',[255/255,178/255,102/255],'LineWidth',line_width) %middle mcp to middle pip
    plot([coord_pixel(1,7),coord_pixel(1,8)],[coord_pixel(2,7),coord_pixel(2,8)],'Color',[255/255,153/255,51/255],'LineWidth',line_width) %middle pip to middle dip
    plot([coord_pixel(1,8),coord_pixel(1,9)],[coord_pixel(2,8),coord_pixel(2,9)],'Color',[255/255,128/255,0/255],'LineWidth',line_width) %middle dip to middle tip

    plot([coord_pixel(1,1),coord_pixel(1,10)],[coord_pixel(2,1),coord_pixel(2,10)],'Color',[153/255,204/255,255/255],'LineWidth',line_width) %wrist to ring mcp
    plot([coord_pixel(1,10),coord_pixel(1,11)],[coord_pixel(2,10),coord_pixel(2,11)],'Color',[102/255,178/255,255/255],'LineWidth',line_width) %ring mcp to ring pip
    plot([coord_pixel(1,11),coord_pixel(1,12)],[coord_pixel(2,11),coord_pixel(2,12)],'Color',[51/255,153/255,255/255],'LineWidth',line_width) %ring pip to ring dip
    plot([coord_pixel(1,12),coord_pixel(1,13)],[coord_pixel(2,12),coord_pixel(2,13)],'Color',[0/255,128/255,255/255],'LineWidth',line_width) %ring dip to ring tip

    plot([coord_pixel(1,1),coord_pixel(1,14)],[coord_pixel(2,1),coord_pixel(2,14)],'Color',[255/255,153/255,255/255],'LineWidth',line_width) %wrist to little mcp
    plot([coord_pixel(1,14),coord_pixel(1,15)],[coord_pixel(2,14),coord_pixel(2,15)],'Color',[255/255,102/255,255/255],'LineWidth',line_width) %little mcp to little pip
    plot([coord_pixel(1,15),coord_pixel(1,16)],[coord_pixel(2,15),coord_pixel(2,16)],'Color',[255/255,51/255,255/255],'LineWidth',line_width) %little pip to little dip
    plot([coord_pixel(1,16),coord_pixel(1,17)],[coord_pixel(2,16),coord_pixel(2,17)],'Color',[255/255,0/255,255/255],'LineWidth',line_width) %little dip to little tip
    

    colorList = [
    230/255 230/255 0/255;
    
    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    0/255 255/255 0/255;

    255/255 204/255 153/255;
    255/255 178/255 102/255;
    255/255 153/255 51/255;
    255/255 128/255 0/255;
    
    153/255 204/255 255/255;
    102/255 178/255 255/255;
    51/255 153/255 255/255;
    0/255 128/255 255/255;
    
    255/255 153/255 255/255;
    255/255 102/255 255/255;
    255/255 51/255 255/255;
    255/255 0/255 255/255;
    
    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;
    255/255 0/255 0/255;
    ];

    scatter(coord_pixel(1,:),coord_pixel(2,:),100,colorList,'filled');

    set(gca,'Units','normalized','Position',[0 0 1 1]);  %# Modify axes size
    set(gcf,'Units','pixels','Position',[200 200 2*imgWidth 2*imgHeight]);  %# Modify figure size

    frame = getframe(gcf);
    framedata = frame.cdata;
    
    xmin = min(coord_pixel(1,:));
    xmax = max(coord_pixel(1,:));
    ymin = min(coord_pixel(2,:));
    ymax = max(coord_pixel(2,:));

    len = max(xmax-xmin+1,ymax-ymin+1) + 70;
    xcenter = (xmin + xmax)/2;
    ycenter = (ymin + ymax)/2;
    
    xmin = max(round(xcenter - len/2),1);
    xmax = min(round(xmin + len),imgWidth);
    ymin = max(round(ycenter - len/2),1);
    ymax = min(round(ymin + len),imgHeight);
    
    framedata = framedata(2*ymin:2*ymax,2*xmin:2*xmax,:);

    hold off;
    close(f); 

    d2img = framedata;
 
end


function f = draw_3Dskeleton_MSRA(d2img,coord_world)
 
    cubicSz = 150;
    coord_world = squeeze(coord_world);

    x = coord_world(1,:);
    y = coord_world(2,:);
    z = coord_world(3,:);
    coord_world(1,:) = -z;
    coord_world(2,:) = x;
    coord_world(3,:) = y;
    line_width = 8;

    f = figure('Position',[100 100 600 450]);
    set(f, 'visible', 'off');
    hold on;
    grid on;

    plot3([coord_world(1,1),coord_world(1,18)],[coord_world(2,1),coord_world(2,18)],[coord_world(3,1),coord_world(3,18)],'Color',[255/255,153/255,153/255],'LineWidth',line_width) %wrist to thumb mcp
    plot3([coord_world(1,18),coord_world(1,19)],[coord_world(2,18),coord_world(2,19)],[coord_world(3,18),coord_world(3,19)],'Color',[255/255,102/255,102/255],'LineWidth',line_width) %thumb mcp to thumb pip
    plot3([coord_world(1,19),coord_world(1,20)],[coord_world(2,19),coord_world(2,20)],[coord_world(3,19),coord_world(3,20)],'Color',[255/255,51/255,51/255],'LineWidth',line_width) %thumb pip to thumb dip
    plot3([coord_world(1,20),coord_world(1,21)],[coord_world(2,20),coord_world(2,21)],[coord_world(3,20),coord_world(3,21)],'Color',[255/255,0/255,0/255],'LineWidth',line_width) %thumb dip to thumb tip

    plot3([coord_world(1,1),coord_world(1,2)],[coord_world(2,1),coord_world(2,2)],[coord_world(3,1),coord_world(3,2)],'Color',[153/255,255/255,153/255],'LineWidth',line_width) %wrist to index mcp
    plot3([coord_world(1,2),coord_world(1,3)],[coord_world(2,2),coord_world(2,3)],[coord_world(3,2),coord_world(3,3)],'Color',[102/255,255/255,102/255],'LineWidth',line_width) %index mcp to index pip
    plot3([coord_world(1,3),coord_world(1,4)],[coord_world(2,3),coord_world(2,4)],[coord_world(3,3),coord_world(3,4)],'Color',[51/255,255/255,51/255],'LineWidth',line_width) %index pip to index dip
    plot3([coord_world(1,4),coord_world(1,5)],[coord_world(2,4),coord_world(2,5)],[coord_world(3,4),coord_world(3,5)],'Color',[0/255,255/255,0/255],'LineWidth',line_width) %index dip to index tip

    plot3([coord_world(1,1),coord_world(1,6)],[coord_world(2,1),coord_world(2,6)],[coord_world(3,1),coord_world(3,6)],'Color',[255/255,204/255,153/255],'LineWidth',line_width) %wrist to middle mcp
    plot3([coord_world(1,6),coord_world(1,7)],[coord_world(2,6),coord_world(2,7)],[coord_world(3,6),coord_world(3,7)],'Color',[255/255,178/255,102/255],'LineWidth',line_width) %middle mcp to middle pip
    plot3([coord_world(1,7),coord_world(1,8)],[coord_world(2,7),coord_world(2,8)],[coord_world(3,7),coord_world(3,8)],'Color',[255/255,153/255,51/255],'LineWidth',line_width) %middle pip to middle dip
    plot3([coord_world(1,8),coord_world(1,9)],[coord_world(2,8),coord_world(2,9)],[coord_world(3,8),coord_world(3,9)],'Color',[255/255,128/255,0/255],'LineWidth',line_width) %middle dip to middle tip

    plot3([coord_world(1,1),coord_world(1,10)],[coord_world(2,1),coord_world(2,10)],[coord_world(3,1),coord_world(3,10)],'Color',[153/255,204/255,255/255],'LineWidth',line_width) %wrist to ring mcp
    plot3([coord_world(1,10),coord_world(1,11)],[coord_world(2,10),coord_world(2,11)],[coord_world(3,10),coord_world(3,11)],'Color',[102/255,178/255,255/255],'LineWidth',line_width) %ring mcp to ring pip
    plot3([coord_world(1,11),coord_world(1,12)],[coord_world(2,11),coord_world(2,12)],[coord_world(3,11),coord_world(3,12)],'Color',[51/255,153/255,255/255],'LineWidth',line_width) %ring pip to ring dip
    plot3([coord_world(1,12),coord_world(1,13)],[coord_world(2,12),coord_world(2,13)],[coord_world(3,12),coord_world(3,13)],'Color',[0/255,128/255,255/255],'LineWidth',line_width) %ring dip to ring tip

    plot3([coord_world(1,1),coord_world(1,14)],[coord_world(2,1),coord_world(2,14)],[coord_world(3,1),coord_world(3,14)],'Color',[255/255,153/255,255/255],'LineWidth',line_width) %wrist to little mcp
    plot3([coord_world(1,14),coord_world(1,15)],[coord_world(2,14),coord_world(2,15)],[coord_world(3,14),coord_world(3,15)],'Color',[255/255,102/255,255/255],'LineWidth',line_width) %little mcp to little pip
    plot3([coord_world(1,15),coord_world(1,16)],[coord_world(2,15),coord_world(2,16)],[coord_world(3,15),coord_world(3,16)],'Color',[255/255,51/255,255/255],'LineWidth',line_width) %little pip to little dip
    plot3([coord_world(1,16),coord_world(1,17)],[coord_world(2,16),coord_world(2,17)],[coord_world(3,16),coord_world(3,17)],'Color',[255/255,0/255,255/255],'LineWidth',line_width) %little dip to little tip
    

    colorList = [
    230/255 230/255 0/255;
    
    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    0/255 255/255 0/255;

    255/255 204/255 153/255;
    255/255 178/255 102/255;
    255/255 153/255 51/255;
    255/255 128/255 0/255;
    
    153/255 204/255 255/255;
    102/255 178/255 255/255;
    51/255 153/255 255/255;
    0/255 128/255 255/255;
    
    255/255 153/255 255/255;
    255/255 102/255 255/255;
    255/255 51/255 255/255;
    255/255 0/255 255/255;
    
    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;
    255/255 0/255 0/255;
    ];
    scatter3(coord_world(1,:),coord_world(2,:),coord_world(3,:),150,colorList,'filled');
    
    set(gca, 'color', [255/255 255/255 255/255])

    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'ZTickLabel',[]);

    
    xcenter = (min(coord_world(1,:)) + max(coord_world(1,:)))/2;
    ycenter = (min(coord_world(2,:)) + max(coord_world(2,:)))/2;
    zcenter = (min(coord_world(3,:)) + max(coord_world(3,:)))/2;
    
    xmin = xcenter - 3*cubicSz;
    xmax = max(coord_world(1,:));
    ymin = min(ycenter-cubicSz/2, min(coord_world(2,:)));
    ymax = max(coord_world(2,:))+30;
    zmin = min(zcenter-cubicSz/2, min(coord_world(3,:)));
    zmax = max(zcenter+cubicSz/2, max(coord_world(3,:)));
    
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    zlim([zmin zmax]);
    
    h_img = surf([xmin;xmin],[ymin ymax;ymin ymax],[zmax zmax;zmin zmin],'CData',d2img,'FaceColor','texturemap');
    set(h_img,'edgecolor',[96/255 96/255 96/255]);
    
    view(62,7);

end




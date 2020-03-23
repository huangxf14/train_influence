load digit_influ_new.mat
y1 = smoothdata(accuracy_array(1:2:end-1));
load digit_ada_theta6.mat
y2 = smoothda'ta(accuracy_array(1:2:end));
y1(1)=y2(1);
x=(1:size(y2,2))*10*(4*4*2)*2;
plot(x,y1,'-',x,y2,'--','LineWidth',1.5);
xlim([0,x(end)+x(end)-x(end-1)])
xlabel('Time slots')
ylabel('Accuracy')
legend({'Influence function','training loss'},'FontSize',15,'Location','best')
set(gca,'FontSize',15);
saveas(gcf,'importance-downsample.jpg')

load device_filter.mat
y1 = smoothdata(accuracy_array(1:end-1));
load device_i_v_2_14.mat
y2 = smoothdata(accuracy_array(1:ceil(end/2)));
y1(1)=y2(1);
x=(1:size(y2,2))*10*4*8;
plot(x,y1,'-',x,y2,'--','LineWidth',1.5);
xlim([0,x(end)+x(end)-x(end-1)])
xlabel('Time slots')
ylabel('Accuracy')
% legend('Without data filtering','With data filtering','Location','best')
legend({'Influence function','training loss'},'FontSize',15,'Location','best')
set(gca,'FontSize',15);
saveas(gcf,'data-select.jpg','jpg')


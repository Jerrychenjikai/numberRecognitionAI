def cmp(a,b):
       global summ
       summ+=1
       return (a<b)
def quicksort(a,start,end,cmp=cmp):
       if start<end:
              x=a[int((start+end)/2)]
              j=end+1
              i=start-1
              while j>i:
                     j-=1
                     while cmp(x,a[j]):
                            j-=1
                     i+=1
                     while cmp(a[i],x):
                            i+=1
                     if i<j:
                            c=a[j]
                            a[j]=a[i]
                            a[i]=c
              quicksort(a,start,j,cmp)
              quicksort(a,j+1,end,cmp)
def bubblesort(a,cmp=cmp):
       for i in range(len(a)-1):
              for j in range(len(a)-1):
                     if not cmp(a[j],a[j+1]):
                            c=a[j]
                            a[j]=a[j+1]
                            a[j+1]=c

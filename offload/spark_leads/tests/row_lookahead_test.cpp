#include "row_lookahead.h"
#include <cassert>
#include <numeric>
int main() {
  x1::RowLookahead c;
  c.replace({2,2,3}, 100);
  auto v=c.version(); auto ids=c.take(); assert((ids==std::vector<int64_t>{2,3}));
  std::vector<uint8_t> w(8192*256,7), s(8192*8,9), ow(8192*256,0), os(8192*8,0);
  assert(c.publish(v,ids,w.data(),s.data()));
  int64_t want[]={3,2,3};
  assert(c.copy(want,ow.data(),os.data(),3)); assert(ow[767]==7 && os[23]==9);
  int64_t miss[]={2,4}; ow[0]=0; assert(!c.copy(miss,ow.data(),os.data(),2)); assert(ow[0]==0);
  std::vector<int64_t> missing; std::vector<size_t> indices;
  c.split(miss,ow.data(),os.data(),2,missing,indices);
  assert(ow[0]==7 && missing==std::vector<int64_t>{4} && indices==std::vector<size_t>{1});
  c.replace({4},100); v=c.version(); ids=c.take(); c.replace({},100); // cancellation
  assert(!c.publish(v,ids,w.data(),s.data())); assert(c.size()==2);
  bool bad=false; try { c.replace({100},100); } catch(const std::invalid_argument&) { bad=true; } assert(bad);
  std::vector<int64_t> full(8192); std::iota(full.begin(),full.end(),0);
  c.replace(full,10000); v=c.version();
  while(c.pending()) { ids=c.take(); assert(ids.size()<=512); assert(c.publish(v,ids,w.data(),s.data())); }
  assert(c.size()==8192);
  c.replace({9000},10000); ids=c.take(); assert(c.publish(c.version(),ids,w.data(),s.data()));
  assert(c.size()==8192); c.clear(); assert(c.size()==0 && !c.pending());
}

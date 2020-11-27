/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
*/

/*
This file is part of diffmicro.

    Diffmicro is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Diffmicro is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Diffmicro.  If not, see <https://www.gnu.org/licenses/>.
*/



#include "stdafx.h"

#include "prep_vet.h"

INDEX prep_vet_del(std::vector<INDEX> &vet, INDEX dim_extern_vet)
{

 INDEX ii,vetii;
 // preprocessing aimed to have a vector of index to delete with the following properties:
// 1 - sorted
// 2 - no repetition inside
// 3 - really existing index to delete
 if (vet.size() == 0) return (0);


 sort(vet.begin(), vet.end());
 // control
 for(ii=0; ii<vet.size(); ++ii)
  {
   check_conditions:
   vetii = vet[ii];

   // pay attention the vector is sorted and we are reading from the beginning!!!
   if(vet[ii-1]==vetii)
    {
     std::cerr <<"error in prep_vet_del --> repetition of n. "
               <<vetii<<std::endl;
     if(del_vet_element(vet, ii)==false) goto break_ii_cycle;
     goto check_conditions;
    }

   if( (vetii < 0) || ( vetii >= dim_extern_vet) )
    {
     std::cerr <<"error in prep_vet_del invalid index --->  0 < (element = "
               <<vetii<<" ) < (dim_extern_vet = "<<dim_extern_vet<<" ) "<<std::endl;
     if(del_vet_element(vet, ii)==false) goto break_ii_cycle;
     goto check_conditions;
    }
//   if()
  }
 break_ii_cycle:

 return (vet.size());
}

void look_up_table_del(INDEX dimold, std::vector<INDEX> &del, std::vector<INDEX> &lut)
{
 // pay attention!!! del vet is going to be modified
 // preparing new adress in the vector to store old adress
 // -1 is for the elements present in del vector
 INDEX dimdel;
 INDEX dimdelm1;
 INDEX i,j,k,to_del;

 dimdel = prep_vet_del(del,dimold);
 dimdelm1 = dimdel-1;

 lut.resize(dimold);

 if(dimdel==0)
  {
   for(i=0; i<dimold; ++i) lut[i] = i;
   return;
  }

 i=0;
 j=0;
 k=0;
 to_del = del[k];
 while(i<dimold)
  {
   if(i != to_del)
    {
     lut[i] = j;
     ++j;
    }
   else
    {
     lut[i] = -1;
     if(k < dimdelm1)
      {
       ++k;
       to_del = del[k];
      }
    }

   ++i;
  }

}


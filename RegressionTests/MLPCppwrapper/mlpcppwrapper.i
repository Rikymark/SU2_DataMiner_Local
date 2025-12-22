/* file : gfg.i */

/* name of module to use*/
%module(docstring="mlcpp wrapper module")
 mlpcppwrapper
%{
	/* Every thing in this file is being copied in 
	 wrapper file. We include the C header file necessary
	 to compile the interface */
	#include "mlpcppwrapper.hpp"

	/* variable declaration*/
%}
%include "std_string.i"
%include "std_vector.i"
%include "typemaps.i"
namespace std {
	%template() vector<string>;
	%template() vector<double>;
	%template() vector<vector<double>>;
}
/* explicitly list functions and variables to be interfaced */
%include "mlpcppwrapper.hpp"
/* or if we want to interface all functions then we can simply
   include header file like this - 
   %include "gfg.h"
*/